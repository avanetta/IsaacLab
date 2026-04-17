#!/usr/bin/env python3
"""
Diffusion Policy V4/V5 Inference for Franka Peg-in-Hole

Key differences from V1-V3:
    - Proprioception: ee_pose(7) + ee_vel(6) + dfdt(3) instead of pose + tau_ext
    - Velocity and dF/dt computed at 1kHz, lowpass filtered (14Hz Butterworth),
      sampled at 30Hz into rolling buffer of 5 timesteps
    - Quaternion sign continuity enforced
    - Action dim is always 7 (x, y, z, qx, qy, qz, qw)

Usage:
    python3 run_diffusion_policy_v4_inference.py \
        --checkpoint /path/to/best_model.pt \
        --stiffness 300 300 300 30 30 20
"""

import os
import sys
import argparse
import time
import math
import csv
import signal
import atexit
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import torch
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from franka_msgs.msg import FrankaRobotState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter_zi, lfilter

sys.path.append(str(Path.home() / "mujoco_lab"))
from diffusion_policy_v4 import DiffusionPolicyV4

# ========================== CONFIGURATION ==========================

CONTROL_FREQ = 30.0        # Hz — policy control loop
ROBOT_FREQ = 1000.0        # Hz — robot state callback rate
FILTER_CUTOFF = 14.0       # Hz — Butterworth lowpass cutoff
FILTER_ORDER = 2
HISTORY_LEN = 5            # Timesteps of proprio history
ACTION_INTEGRATION_STEPS = 4  # Accumulate deltas over N steps (30Hz / 4 = 7.5Hz effective)

# --- Temporal ensembling weighting ---
# 'uniform': all overlapping plans weighted equally
# 'exponential': newer predictions weighted more (w = exp(-idx / AGG_DECAY))
AGG_WEIGHTS = 'exponential'
AGG_DECAY = 8.0  # lower = more aggressive decay of old predictions

# --- Action EMA smoothing (applied after ensembling + integration) ---
ACTION_EMA_ENABLED = True
ACTION_EMA_ALPHA = 0.4   # ~3Hz cutoff at 30Hz. Lower = smoother. Set to 1.0 to disable.

# --- Hole detection ---
HOLE_DETECTION_METHOD = 'pinhole'  # 'pinhole' or 'pointcloud'
HOLE_Z_ROBOT_FRAME = 0.027
# DETECTION_OFFSET = np.array([0.0001, 0.0082])  # [dx, dy] meters, old calibration (2026-03-23)
DETECTION_OFFSET = np.array([0.0059, 0.0103])  # [dx, dy] meters, from calibration (10 samples, 2026-04-07)

# --- Heuristic gripper release — set GRIPPER_HEURISTIC_ENABLED = False to disable ---
GRIPPER_HEURISTIC_ENABLED = True
GRIPPER_Z_THRESH_M = 0.056        # ee_z in robot frame must be below this (meters)
# ===================================================================






APPROACH_STIFFNESS = [1200, 1200.0, 1500.0, 90.0, 90.0, 65.0]
POLICY_STIFFNESS = [1200.0, 1200.0, 2000.0, 60.0, 60.0, 20.0]


try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except Exception:
    ZED_AVAILABLE = False
    print("ERROR: ZED SDK not available!")
    sys.exit(1)


class GripperReleaseChecker:
    """Heuristic: open gripper when ee_z drops below threshold. To disable: GRIPPER_HEURISTIC_ENABLED = False"""
    def check(self, ee_z_robot):
        if not GRIPPER_HEURISTIC_ENABLED:
            return False
        return ee_z_robot < GRIPPER_Z_THRESH_M


class ImageProcessor:
    """9-channel temporal image stack: [current, previous, difference]."""

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        self.image_history = deque(maxlen=2)

    def preprocess_single_image(self, rgb_image):
        img = cv2.resize(rgb_image, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.mean) / self.std
        return img

    def update(self, rgb_image):
        processed = self.preprocess_single_image(rgb_image)
        self.image_history.append(processed)
        if len(self.image_history) < 2:
            self.image_history.append(processed)
        current = self.image_history[-1]
        previous = self.image_history[-2]
        diff = current - previous
        return np.concatenate([current, previous, diff], axis=0)


class ButterworthFilter:
    """Causal Butterworth lowpass filter for real-time 1kHz signal processing."""

    def __init__(self, cutoff, fs, order, n_channels):
        self.b, self.a = butter(order, cutoff / (0.5 * fs), btype='low')
        # One filter state per channel
        zi_single = lfilter_zi(self.b, self.a)
        self.zi = np.tile(zi_single, (n_channels, 1))
        self.initialized = False

    def reset(self, initial_value):
        """Reset filter state to steady-state for given initial value."""
        zi_single = lfilter_zi(self.b, self.a)
        for ch in range(len(initial_value)):
            self.zi[ch] = zi_single * initial_value[ch]
        self.initialized = True

    def __call__(self, x):
        """Filter a single sample x (1D array). Returns filtered value."""
        out = np.empty_like(x)
        for ch in range(len(x)):
            y, self.zi[ch] = lfilter(self.b, self.a, [x[ch]], zi=self.zi[ch])
            out[ch] = y[0]
        return out


class DiffusionPolicyV4InferenceNode(Node):
    """ROS2 node for V4/V5 Diffusion Policy inference with force derivative proprioception."""

    def __init__(self, args):
        super().__init__('diffusion_policy_v4_inference_node')
        self.args = args

        # Hole position (set by hole detection)
        self.hole_position = None

        # Robot state
        self.current_robot_state = None
        self.current_ee_pose_robot_frame = None  # [x, y, z, qx, qy, qz, qw]

        # ── 1kHz signal processing state ──
        # Previous values for finite difference
        self.prev_ee_pose = None       # [x,y,z,qx,qy,qz,qw]
        self.prev_f_ext = None         # [fx, fy, fz]
        self.prev_quat_sign = None     # For quaternion sign continuity

        # Butterworth filters (initialized on first callback)
        self.vel_filter = ButterworthFilter(FILTER_CUTOFF, ROBOT_FREQ, FILTER_ORDER, 6)
        self.dfdt_filter = ButterworthFilter(FILTER_CUTOFF, ROBOT_FREQ, FILTER_ORDER, 3)

        # Latest filtered values (written at 1kHz, read at 30Hz)
        self.latest_ee_pose = None     # [7] in hole frame, sign-continuous quat
        self.latest_ee_vel = None      # [6] filtered
        self.latest_dfdt = None        # [3] filtered

        # History buffers (filled at 30Hz)
        self.pose_history = deque(maxlen=HISTORY_LEN)
        self.vel_history = deque(maxlen=HISTORY_LEN)
        self.dfdt_history = deque(maxlen=HISTORY_LEN)

        # Baseline force (for baseline compensation of dF/dt)
        self.f_ext_baseline = None

        # Image processor
        self.image_processor = ImageProcessor()

        # Camera calibration
        self.T_cam_to_ee = None
        self.load_camera_calibration()

        # ZED camera — open at 1080p for hole detection, reopen at 720p for inference
        self.zed = sl.Camera()
        self.zed_image = sl.Mat()
        self._open_zed(sl.RESOLUTION.HD1080)

        # Receding horizon
        self.action_horizon = 64
        self.execution_horizon = args.execution_horizon

        # Temporal ensembling: average overlapping predictions from consecutive plans
        # Each cell in the queue holds (action_buffer [64,7], global_step_offset)
        self.plan_queue = deque()  # list of (actions [64,7], start_step)
        self.global_step = 0
        self.steps_since_replan = 0

        # Action integration: accumulate position deltas, compose rotation deltas
        self.last_prediction = None
        self.accumulated_pos_delta = np.zeros(3)
        self.accumulated_rot_delta = R.identity()
        self.ema_action = None  # Action EMA state

        # Heuristic gripper release
        self.gripper_checker = GripperReleaseChecker()
        self.gripper_opened = False
        self.gripper_client = None  # lazy init

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Device: {self.device}")
        self.model = self.load_model(args.checkpoint)
        self.get_logger().info(f"Model loaded (exec_horizon={self.execution_horizon})")

        # ROS2 setup
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.robot_state_sub = self.create_subscription(
            FrankaRobotState,
            "/franka_robot_state_broadcaster/robot_state",
            self.robot_state_callback,
            qos_profile
        )
        self.pose_command_pub = self.create_publisher(
            Float64MultiArray,
            "/cartesian_position_controller/commands",
            10
        )

        # Control loop timer (30Hz)
        self.control_timer = self.create_timer(1.0 / CONTROL_FREQ, self.control_loop)
        self.control_active = False

        # Performance metrics
        self.inference_times = []

        # Stiffness: approach uses high stiffness, policy uses --stiffness
        self.current_K = list(APPROACH_STIFFNESS)
        self.policy_K = list(args.stiffness) if args.stiffness is not None else list(POLICY_STIFFNESS)

        # Logging
        self.setup_logging()

        self.get_logger().info("V4 inference node initialized")

    # ─────────────────────── Model Loading ───────────────────────

    def _open_zed(self, resolution):
        """Open (or reopen) ZED camera at given resolution."""
        if self.zed.is_opened():
            self.zed.close()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {err}")
            sys.exit(1)
        self.get_logger().info(f"ZED camera opened at {resolution}")

        # Cache intrinsics for pinhole detection
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.cam_fx = calib.fx
        self.cam_fy = calib.fy
        self.cam_cx = calib.cx
        self.cam_cy = calib.cy

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model = DiffusionPolicyV4(
            action_horizon=self.action_horizon,
            action_dim=7,
            diffusion_steps=100,
            ddim_steps=10,
        )
        model = model.to(self.device)

        self.get_logger().info("Loading model weights")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Apply EMA shadow weights if available
        if 'ema_shadow' in checkpoint:
            self.get_logger().info("Applying EMA shadow weights")
            ema_shadow = checkpoint['ema_shadow']
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ema_shadow:
                        param.copy_(ema_shadow[name].to(param.device))
        else:
            self.get_logger().warn("No EMA shadow found, using regular weights")

        model.eval()
        return model

    # ─────────────────────── Camera ───────────────────────

    def load_camera_calibration(self):
        calib_base = Path("/home/pdzuser/emre_data/camera_calibration")
        # Use most recent calibration folder
        rec_folders = sorted(
            [p for p in calib_base.iterdir()
             if p.is_dir() and p.name.startswith(("Rec_", "calibration_"))],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not rec_folders:
            self.get_logger().error("No camera calibration found!")
            sys.exit(1)

        calib_file = rec_folders[0] / "calib" / "info" / "robot_transforms.yaml"
        with open(calib_file) as f:
            calib_data = yaml.safe_load(f)
        self.T_cam_to_ee = np.array(calib_data['robot_to_camera'])
        self.get_logger().info(f"Calibration loaded from: {rec_folders[0].name}")

    def capture_camera_frame(self):
        err = self.zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            img_bgr = self.zed_image.get_data()[:, :, :3].copy()
            return np.ascontiguousarray(img_bgr, dtype=np.uint8)
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    # ─────────────────────── Hole Detection ───────────────────────

    def detect_holes_in_image(self, img_bgr, point_cloud=None, ee_pose=None):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        holes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 5000:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            pos_robot = None
            if ee_pose is not None:
                if HOLE_DETECTION_METHOD == 'pinhole':
                    ray_cam = np.array([(cx - self.cam_cx) / self.cam_fx,
                                        (cy - self.cam_cy) / self.cam_fy,
                                        1.0, 0.0])
                    origin_cam = np.array([0.0, 0.0, 0.0, 1.0])
                    T = np.eye(4)
                    T[:3, :3] = R.from_quat(ee_pose[3:7]).as_matrix()
                    T[:3, 3] = ee_pose[:3]
                    T_cam_to_base = T @ self.T_cam_to_ee
                    ray_base = (T_cam_to_base @ ray_cam)[:3]
                    origin_base = (T_cam_to_base @ origin_cam)[:3]
                    if abs(ray_base[2]) > 1e-6:
                        t = (HOLE_Z_ROBOT_FRAME - origin_base[2]) / ray_base[2]
                        if t > 0:
                            pos_robot = origin_base + t * ray_base
                elif point_cloud is not None:
                    err, point3D = point_cloud.get_value(cx, cy)
                    if err == sl.ERROR_CODE.SUCCESS and not np.isnan(point3D[0]):
                        point_cam = np.array([point3D[0], point3D[1], point3D[2], 1.0])
                        point_ee = self.T_cam_to_ee @ point_cam
                        T_ee_to_base = np.eye(4)
                        T_ee_to_base[:3, :3] = R.from_quat(ee_pose[3:7]).as_matrix()
                        T_ee_to_base[:3, 3] = ee_pose[:3]
                        pos_robot = (T_ee_to_base @ point_ee)[:3]
            holes.append(((cx, cy), pos_robot))
        return holes

    def detect_hole(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("HOLE DETECTION - Click on hole center, 'q' to cancel")
        self.get_logger().info("=" * 60)

        cv2.namedWindow('Hole Detection')
        pending_click = {'uv': None}

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_click['uv'] = (x, y)

        cv2.setMouseCallback('Hole Detection', mouse_cb)
        point_cloud = sl.Mat()

        while True:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
                continue

            captured_ee_pose = (self.current_ee_pose_robot_frame.copy()
                                if self.current_ee_pose_robot_frame is not None else None)

            self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            img_bgr = self.zed_image.get_data()[:, :, :3].copy()
            img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)

            holes = self.detect_holes_in_image(img_bgr, point_cloud, ee_pose=captured_ee_pose)
            for (center, _) in holes:
                cv2.circle(img_bgr, center, 4, (200, 200, 200), -1)

            if pending_click['uv'] is not None and holes:
                px, py = pending_click['uv']
                valid = [(h, p) for (h, p) in holes if p is not None]
                if not valid:
                    self.get_logger().error("No holes with valid 3D. Try again.")
                    pending_click['uv'] = None
                    continue

                dists = [((h[0]-px)**2 + (h[1]-py)**2, h, p) for (h, p) in valid]
                dists.sort(key=lambda t: t[0])
                nearest_pixel, hole_pos = dists[0][1], dists[0][2]

                cv2.circle(img_bgr, nearest_pixel, 8, (0, 255, 0), 2)
                cv2.imshow('Hole Detection', img_bgr)
                cv2.waitKey(1)

                hole_pos[2] = HOLE_Z_ROBOT_FRAME
                hole_pos[:2] += DETECTION_OFFSET
                self.get_logger().info(
                    f"Hole at: [{hole_pos[0]*1000:.1f}, {hole_pos[1]*1000:.1f}, {hole_pos[2]*1000:.1f}] mm (offset applied)")

                response = input("Correct? (y/n): ").strip().lower()
                if response == 'y':
                    self.hole_position = hole_pos
                    cv2.destroyWindow('Hole Detection')
                    return True
                pending_click['uv'] = None

            elif pending_click['uv'] is not None:
                self.get_logger().error("No holes detected. Try again.")
                pending_click['uv'] = None

            cv2.imshow('Hole Detection', img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('Hole Detection')
                return False

    # ─────────────────────── Hole Frame Transform ───────────────────────

    def get_ee_pose_hole_frame(self):
        """Transform EE pose to hole frame. Returns [x,y,z,qx,qy,qz,qw]."""
        if self.current_ee_pose_robot_frame is None:
            return np.zeros(7)
        pos_hole = self.current_ee_pose_robot_frame[:3] - self.hole_position
        quat = self.current_ee_pose_robot_frame[3:].copy()
        return np.concatenate([pos_hole, quat])

    def enforce_quat_continuity(self, quat):
        """Flip quaternion sign if it diverges from previous to maintain continuity."""
        if self.prev_quat_sign is not None:
            if np.dot(quat, self.prev_quat_sign) < 0:
                quat = -quat
        self.prev_quat_sign = quat.copy()
        return quat

    # ─────────────────── 1kHz Robot State Callback ───────────────────

    def robot_state_callback(self, msg: FrankaRobotState):
        """Process robot state at 1kHz: compute velocity + dF/dt, lowpass filter."""
        self.current_robot_state = msg

        pos = msg.o_t_ee.pose.position
        ori = msg.o_t_ee.pose.orientation
        ee_pose_robot = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        self.current_ee_pose_robot_frame = ee_pose_robot

        # Need hole position to compute hole-frame signals
        if self.hole_position is None:
            return

        # Current pose in hole frame with quaternion sign continuity
        pos_hole = ee_pose_robot[:3] - self.hole_position
        quat = self.enforce_quat_continuity(ee_pose_robot[3:].copy())
        ee_pose_hole = np.concatenate([pos_hole, quat])

        # Current Cartesian external force
        wrench = msg.o_f_ext_hat_k.wrench
        f_ext = np.array([wrench.force.x, wrench.force.y, wrench.force.z])

        # Finite difference for velocity and dF/dt
        dt = 1.0 / ROBOT_FREQ

        if self.prev_ee_pose is not None:
            # Linear velocity
            lin_vel = (pos_hole - self.prev_ee_pose[:3]) / dt

            # Angular velocity from quaternion: omega = 2 * imag(q_curr * conj(q_prev)) / dt
            q_curr = quat  # [qx, qy, qz, qw]
            q_prev = self.prev_ee_pose[3:]
            # q_conj = [-qx, -qy, -qz, qw]
            q_prev_conj = np.array([-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]])
            # Quaternion multiply: q_curr * q_prev_conj
            dq = self._quat_multiply(q_curr, q_prev_conj)
            ang_vel = 2.0 * dq[:3] / dt  # imaginary part

            vel_raw = np.concatenate([lin_vel, ang_vel])

            # dF/dt
            dfdt_raw = (f_ext - self.prev_f_ext) / dt

            # Initialize filters on first valid sample
            if not self.vel_filter.initialized:
                self.vel_filter.reset(vel_raw)
                self.dfdt_filter.reset(dfdt_raw)

            # Apply lowpass filter
            self.latest_ee_vel = self.vel_filter(vel_raw)
            self.latest_dfdt = self.dfdt_filter(dfdt_raw)
        else:
            self.latest_ee_vel = np.zeros(6)
            self.latest_dfdt = np.zeros(3)

        self.latest_ee_pose = ee_pose_hole
        self.prev_ee_pose = ee_pose_hole.copy()
        self.prev_f_ext = f_ext.copy()

    @staticmethod
    def _quat_multiply(q1, q2):
        """Hamilton product of two quaternions [qx, qy, qz, qw]."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])

    # ─────────────────────── Buffer Init ───────────────────────

    def initialize_buffers(self):
        self.get_logger().info("Initializing state buffers...")

        # Wait for robot state
        while self.current_robot_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Wait for 1kHz processing to produce valid signals
        # (need at least 2 callbacks for finite difference)
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.001)

        # Capture 2 image frames
        for i in range(2):
            bgr = self.capture_camera_frame()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            processed = self.image_processor.preprocess_single_image(rgb)
            self.image_processor.image_history.append(processed)
            if i < 1:
                time.sleep(0.033)

        # Fill history with current state (5 copies)
        pose = self.latest_ee_pose if self.latest_ee_pose is not None else np.zeros(7)
        vel = self.latest_ee_vel if self.latest_ee_vel is not None else np.zeros(6)
        dfdt = self.latest_dfdt if self.latest_dfdt is not None else np.zeros(3)

        for _ in range(HISTORY_LEN):
            self.pose_history.append(pose.copy())
            self.vel_history.append(vel.copy())
            self.dfdt_history.append(dfdt.copy())

        self.get_logger().info(f"Buffers initialized. Pose: {pose[:3]*1000} mm")

    # ─────────────────────── Observation ───────────────────────

    def create_observation(self):
        if len(self.pose_history) < HISTORY_LEN:
            return None

        # Capture and process image
        bgr = self.capture_camera_frame()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_stack = self.image_processor.update(rgb)

        # Sample latest filtered signals into history
        if self.latest_ee_pose is not None:
            self.pose_history.append(self.latest_ee_pose.copy())
        if self.latest_ee_vel is not None:
            self.vel_history.append(self.latest_ee_vel.copy())
        if self.latest_dfdt is not None:
            self.dfdt_history.append(self.latest_dfdt.copy())

        # Stack history
        pose_hist = np.stack(list(self.pose_history), axis=0)   # [5, 7]
        vel_hist = np.stack(list(self.vel_history), axis=0)     # [5, 6]
        dfdt_hist = np.stack(list(self.dfdt_history), axis=0)   # [5, 3]

        # To tensors (raw — model normalizes internally)
        obs = {
            'img_stack': torch.from_numpy(img_stack).float().unsqueeze(0).to(self.device),
            'ee_pose_history': torch.from_numpy(pose_hist).float().unsqueeze(0).to(self.device),
            'ee_vel_history': torch.from_numpy(vel_hist).float().unsqueeze(0).to(self.device),
            'dfdt_history': torch.from_numpy(dfdt_hist).float().unsqueeze(0).to(self.device),
        }
        return obs

    # ─────────────────────── Approach Motion ───────────────────────

    def move_to_approach_position(self, height_above_hole=0.06):
        from scipy.spatial.transform import Slerp

        self.get_logger().info(f"Moving to approach position ({height_above_hole*100:.0f}cm above hole)...")

        current_pos = self.current_ee_pose_robot_frame[:3].copy()
        current_quat = self.current_ee_pose_robot_frame[3:].copy()

        target_pos = self.hole_position.copy()
        target_pos[2] += height_above_hole

        # Orientation: straight down
        downward_rot = R.from_rotvec([np.pi, 0, 0])
        target_quat = downward_rot.as_quat()

        # SLERP for smooth orientation
        current_rot = R.from_quat(current_quat)
        target_rot = R.from_quat(target_quat)
        slerp = Slerp([0, 1], R.concatenate([current_rot, target_rot]))

        distance = np.linalg.norm(target_pos - current_pos)
        num_steps = min(400, max(100, int(distance / 0.001)))
        direction = (target_pos - current_pos) / distance if distance > 0 else np.zeros(3)

        self.get_logger().info(f"Distance: {distance*1000:.1f}mm, steps: {num_steps}")

        for i in range(num_steps):
            t = (i + 1) / num_steps
            s = 0.5 * (1 - np.cos(np.pi * t))

            interp_pos = current_pos + direction * distance * s
            interp_quat = slerp(s).as_quat()

            msg = Float64MultiArray()
            msg.data = [float(v) for v in interp_pos] + [float(v) for v in interp_quat] + [float(k) for k in self.current_K]

            for _ in range(2):
                self.pose_command_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(1.0 / 30.0)

        # Hold at final position
        for _ in range(30):
            msg = Float64MultiArray()
            msg.data = [float(v) for v in target_pos] + [float(v) for v in target_quat] + [float(k) for k in self.current_K]
            self.pose_command_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / 30.0)

        final_err = np.linalg.norm(self.current_ee_pose_robot_frame[:3] - target_pos)
        self.get_logger().info(f"Reached approach position (error: {final_err*1000:.1f}mm)")
        return True

    # ─────────────────────── Action Execution ───────────────────────

    def execute_action(self, action):
        """Execute absolute pose action (hole frame) on robot."""
        # Hole frame -> robot frame
        target_pos_robot = action[:3] + self.hole_position

        # Normalize quaternion and ensure closest path to current orientation
        target_quat = action[3:7].copy()
        target_quat = target_quat / np.linalg.norm(target_quat)
        if self.current_ee_pose_robot_frame is not None:
            current_quat = self.current_ee_pose_robot_frame[3:7]
            if np.dot(target_quat, current_quat) < 0:
                target_quat = -target_quat

        msg = Float64MultiArray()
        pose_data = [float(v) for v in target_pos_robot] + [float(v) for v in target_quat]
        if not self.args.energy_node:
            pose_data.extend([float(k) for k in self.current_K])
        msg.data = pose_data

        # Log stiffness on first few calls to verify
        if len(self.inference_times) < 3:
            mode = "pose-only (energy node)" if self.args.energy_node else f"K={self.current_K}"
            # self.get_logger().info(f"[execute_action] {mode}")

        self.pose_command_pub.publish(msg)

    def open_gripper(self):
        """Open gripper via Move action (lazy init)."""
        if self.gripper_client is None:
            from rclpy.action import ActionClient
            from franka_msgs.action import Move
            self.gripper_client = ActionClient(self, Move, '/fr3_gripper/move')
            self.gripper_client.wait_for_server(timeout_sec=5.0)

        from franka_msgs.action import Move
        goal = Move.Goal()
        goal.width = 0.08  # fully open
        goal.speed = 0.1
        self.gripper_client.send_goal_async(goal)
        self.get_logger().info("Gripper OPEN command sent")

    # ─────────────────────── Control Loop ───────────────────────

    @torch.no_grad()
    def control_loop(self):
        if not self.control_active:
            return

        try:
            t_start = time.time()

            obs = self.create_observation()
            if obs is None:
                self.get_logger().error("Failed to create observation!")
                self.control_active = False
                return

            # Debug on first iteration
            if len(self.inference_times) == 0:
                self.get_logger().info("=== FIRST OBSERVATION DEBUG ===")
                pose_raw = np.stack(list(self.pose_history), axis=0)
                vel_raw = np.stack(list(self.vel_history), axis=0)
                dfdt_raw = np.stack(list(self.dfdt_history), axis=0)
                self.get_logger().info(f"Pose [5,7]: last={pose_raw[-1,:3]*1000} mm")
                self.get_logger().info(f"Vel [5,6]: last={vel_raw[-1]} m/s")
                self.get_logger().info(f"dF/dt [5,3]: last={dfdt_raw[-1]} N/s")
                self.get_logger().info(f"Normalizer pose mean: {self.model.normalizer.pose_mean.cpu().numpy()}")
                self.get_logger().info(f"Normalizer vel mean: {self.model.normalizer.vel_mean.cpu().numpy()}")
                self.get_logger().info(f"Normalizer dfdt mean: {self.model.normalizer.dfdt_mean.cpu().numpy()}")
                self.get_logger().info(f"Normalizer action mean: {self.model.normalizer.action_mean.cpu().numpy()}")

            # Replan every execution_horizon steps
            if self.steps_since_replan >= self.execution_horizon or len(self.plan_queue) == 0:
                result = self.model.forward_inference(obs, use_ddim=True)
                new_actions = result['actions'][0].cpu().numpy()  # [64, 7]

                # Enforce quaternion sign continuity within this plan
                for i in range(1, len(new_actions)):
                    if np.dot(new_actions[i, 3:7], new_actions[i-1, 3:7]) < 0:
                        new_actions[i, 3:7] *= -1

                # Also align with previous plan's quaternion sign
                if len(self.plan_queue) > 0:
                    prev_actions = self.plan_queue[-1][0]
                    prev_step = self.global_step - self.plan_queue[-1][1]
                    if prev_step < len(prev_actions):
                        if np.dot(new_actions[0, 3:7], prev_actions[prev_step, 3:7]) < 0:
                            new_actions[:, 3:7] *= -1

                self.plan_queue.append((new_actions, self.global_step))
                self.steps_since_replan = 0

                # Drop old plans that no longer overlap with current step
                while len(self.plan_queue) > 1:
                    oldest_actions, oldest_start = self.plan_queue[0]
                    if self.global_step - oldest_start >= len(oldest_actions):
                        self.plan_queue.popleft()
                    else:
                        break

            # Temporal ensembling: weighted average of overlapping predictions
            pos_sum = np.zeros(3)
            quat_sum = np.zeros(4)
            quat_list = []
            weights = []
            weight_sum = 0.0

            for actions, start_step in self.plan_queue:
                idx = self.global_step - start_step
                if 0 <= idx < len(actions):
                    if AGG_WEIGHTS == 'exponential':
                        w = np.exp(-idx / AGG_DECAY)
                    else:
                        w = 1.0
                    pos_sum += w * actions[idx, :3]
                    quat_list.append(actions[idx, 3:7])
                    weights.append(w)
                    weight_sum += w

            action = np.empty(7)
            action[:3] = pos_sum / weight_sum

            # Weighted quaternion average: align signs, weighted sum, normalize
            ref_quat = quat_list[0]
            for q, w in zip(quat_list, weights):
                if np.dot(q, ref_quat) < 0:
                    quat_sum -= w * q
                else:
                    quat_sum += w * q
            action[3:7] = quat_sum / np.linalg.norm(quat_sum)

            n_plans = len(quat_list)
            self.global_step += 1
            self.steps_since_replan += 1

            # Action integration
            if not self.args.no_integration and self.last_prediction is not None:
                pos_delta = action[:3] - self.last_prediction[:3]
                self.accumulated_pos_delta += pos_delta

                integrated_action = np.empty(7)
                integrated_action[:3] = action[:3] + self.accumulated_pos_delta

                # Rotation integration (--integrate-rotation to enable, off by default)
                if self.args.integrate_rotation:
                    rot_curr = R.from_quat(action[3:7])
                    rot_prev = R.from_quat(self.last_prediction[3:7])
                    rot_delta = rot_curr * rot_prev.inv()
                    self.accumulated_rot_delta = rot_delta * self.accumulated_rot_delta
                    integrated_action[3:7] = (self.accumulated_rot_delta * rot_curr).as_quat()
                else:
                    integrated_action[3:7] = action[3:7]
            else:
                integrated_action = action.copy()
                if not self.args.no_integration:
                    self.accumulated_pos_delta = np.zeros(3)
                    self.accumulated_rot_delta = R.identity()

            self.last_prediction = action.copy()

            # Action EMA smoothing (to disable: set ACTION_EMA_ENABLED = False)
            if ACTION_EMA_ENABLED:
                if self.ema_action is None:
                    self.ema_action = integrated_action.copy()
                else:
                    self.ema_action = ACTION_EMA_ALPHA * integrated_action + (1.0 - ACTION_EMA_ALPHA) * self.ema_action
                    # Re-normalize quaternion after linear blend
                    qnorm = np.linalg.norm(self.ema_action[3:7])
                    if qnorm > 1e-6:
                        self.ema_action[3:7] /= qnorm
                final_action = self.ema_action
            else:
                final_action = integrated_action

            # Per-step logging disabled to avoid communication constraint violations
            # self.get_logger().info(
            #     f"[step {self.global_step-1}, plans={n_plans}] Target: "
            #     f"X={final_action[0]*1000:.2f}mm Y={final_action[1]*1000:.2f}mm Z={final_action[2]*1000:.2f}mm | "
            #     f"quat=[{final_action[3]:.4f}, {final_action[4]:.4f}, {final_action[5]:.4f}, {final_action[6]:.4f}]")

            self.log_step_data(final_action)
            self.execute_action(final_action)

            # Heuristic gripper release check
            if not self.gripper_opened and self.current_ee_pose_robot_frame is not None:
                ee_z = self.current_ee_pose_robot_frame[2]
                if self.gripper_checker.check(ee_z):
                    self.get_logger().info(
                        f"GRIPPER RELEASE triggered (ee_z={ee_z*1000:.1f}mm < {GRIPPER_Z_THRESH_M*1000:.0f}mm)")
                    self.control_active = False
                    self.gripper_opened = True
                    self.open_gripper()

            t_end = time.time()
            inference_time = (t_end - t_start) * 1000
            self.inference_times.append(inference_time)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            import traceback
            traceback.print_exc()
            self.control_active = False

    # ─────────────────────── Logging ───────────────────────

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"v4_run_{timestamp}"
        self.run_dir = Path("/home/pdzuser/emre_data/v4_inference") / self.run_name
        self.frames_dir = self.run_dir / "frames"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

        self.csv_file = self.run_dir / f"trajectory_{self.run_name}.csv"
        self.csv_fp = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_fp)

        header = [
            'timestamp',
            'ee_pos_x', 'ee_pos_y', 'ee_pos_z',
            'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw',
            'ee_vx', 'ee_vy', 'ee_vz', 'ee_wx', 'ee_wy', 'ee_wz',
            'dfdt_x', 'dfdt_y', 'dfdt_z',
            'f_ext_x', 'f_ext_y', 'f_ext_z',
            'cmd_pos_x', 'cmd_pos_y', 'cmd_pos_z',
            'cmd_qx', 'cmd_qy', 'cmd_qz', 'cmd_qw',
            'K_x', 'K_y', 'K_z', 'K_rx', 'K_ry', 'K_rz',
        ]
        self.csv_writer.writerow(header)
        self.csv_fp.flush()
        self.frame_counter = 0
        self.get_logger().info(f"Logging to: {self.run_dir}")

    def log_step_data(self, action):
        try:
            pose = self.latest_ee_pose if self.latest_ee_pose is not None else np.zeros(7)
            vel = self.latest_ee_vel if self.latest_ee_vel is not None else np.zeros(6)
            dfdt = self.latest_dfdt if self.latest_dfdt is not None else np.zeros(3)

            # Current f_ext
            if self.current_robot_state is not None:
                w = self.current_robot_state.o_f_ext_hat_k.wrench
                f_ext = [w.force.x, w.force.y, w.force.z]
            else:
                f_ext = [0.0, 0.0, 0.0]

            row = [
                time.time(),
                *pose.tolist(),
                *vel.tolist(),
                *dfdt.tolist(),
                *f_ext,
                *action[:3].tolist(),
                *action[3:7].tolist(),
                *self.current_K,
            ]
            self.csv_writer.writerow(row)
            self.csv_fp.flush()

            self.frame_counter += 1
        except Exception as e:
            self.get_logger().error(f"Logging error: {e}")

    def cleanup_logging(self):
        try:
            if hasattr(self, 'csv_fp') and self.csv_fp is not None:
                self.csv_fp.close()
                self.get_logger().info(f"Saved trajectory to {self.csv_file}")
                self.get_logger().info(f"Saved {self.frame_counter} frames to {self.frames_dir}")
        except Exception as e:
            self.get_logger().error(f"Cleanup error: {e}")

    # ─────────────────────── Main Run ───────────────────────

    def run(self):
        self.get_logger().info("Starting hole detection...")
        if not self.detect_hole():
            self.get_logger().error("Hole detection failed!")
            return

        # Reopen camera at 720p for inference
        self._open_zed(sl.RESOLUTION.HD720)

        if not self.move_to_approach_position(height_above_hole=0.06):
            self.get_logger().error("Failed to reach approach position!")
            return

        self.initialize_buffers()

        input("\nReady at approach position. Press ENTER to start policy...")

        # Launch energy node or switch to policy stiffness
        self.energy_proc = None
        if self.args.energy_node:
            import subprocess
            energy_script = str(Path(__file__).parent / "energy_stiffness_node.py")
            self.energy_proc = subprocess.Popen(
                ["bash", "-c",
                 f"source ~/franka_ros2_ws/install/setup.bash && python3 {energy_script}"],
            )
            time.sleep(1.0)  # wait for node to start publishing
            self.get_logger().info("Energy stiffness node launched")
        else:
            self.current_K = list(self.policy_K)
            self.get_logger().info(f"Policy stiffness: {self.current_K}")

        self.get_logger().info("Starting policy execution at 30Hz")
        self.control_active = True

        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            self.get_logger().info("Stopping policy execution")

        if self.energy_proc is not None:
            self.energy_proc.terminate()
            self.energy_proc.wait()
            self.get_logger().info("Energy stiffness node stopped")

        if self.inference_times:
            avg = np.mean(self.inference_times)
            mx = np.max(self.inference_times)
            self.get_logger().info(f"Performance: {avg:.1f}ms avg, {mx:.1f}ms max")

        if self.zed is not None:
            self.zed.close()

    def shutdown(self):
        self.control_active = False
        self.cleanup_logging()
        if self.zed is not None:
            self.zed.close()


def main():
    parser = argparse.ArgumentParser(description='Diffusion Policy V4/V5 Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--execution_horizon', type=int, default=16,
                        help='Steps to execute before re-predicting (default: 16)')
    parser.add_argument('--stiffness', type=float, nargs=6, default=None,
                        help='Stiffness [Kx Ky Kz Krx Kry Krz]')
    parser.add_argument('--no-integration', action='store_true',
                        help='Disable action integration')
    parser.add_argument('--integrate-rotation', action='store_true',
                        help='Enable rotation integration (off by default, can cause rz drift)')
    parser.add_argument('--energy-node', action='store_true',
                        help='Use energy-based stiffness controller (pose-only commands)')
    args = parser.parse_args()

    rclpy.init()
    node = DiffusionPolicyV4InferenceNode(args)

    def cleanup_handler(signum=None, frame=None):
        print("\nShutting down...")
        node.cleanup_logging()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(node.cleanup_logging)

    try:
        node.run()
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()