# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to add mimic annotations to demos to be used as source demos for mimic dataset generation.
"""

import argparse
import math

from isaaclab.app import AppLauncher

# Launching Isaac Sim Simulator first.


# add argparse arguments
parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--annotate_subtask_start_signals",
    action="store_true",
    default=False,
    help="Enable annotating start points of subtasks.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed
    # by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

# Only enables inputs if this script is NOT headless mode
if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils import math as PoseUtils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        # Get object poses and convert to 4x4 format
        object_poses = self._env.get_object_poses()
        converted_object_poses = {}
        
        for obj_name, pose_data in object_poses.items():
            # Convert to tensor if needed
            if not isinstance(pose_data, torch.Tensor):
                pose_data = torch.from_numpy(pose_data).to(self._env.device).float()
            else:
                pose_data = pose_data.to(self._env.device).float()
            
            # Remember original dimensionality to match input shape
            original_dim = pose_data.dim()
            
            # Handle 7D pose format: [x, y, z, qw, qx, qy, qz]
            if pose_data.shape[-1] == 7:
                # Ensure batch dimension exists
                if pose_data.dim() == 1:
                    pose_data = pose_data.unsqueeze(0)  # Shape: (1, 7)
                
                pos = pose_data[:, :3]  # Shape: (batch, 3)
                quat = pose_data[:, 3:]  # Shape: (batch, 4)
                
                # Convert quaternion to rotation matrix using matrix_from_quat
                rot_mat = PoseUtils.matrix_from_quat(quat)  # Shape: (batch, 3, 3)
                
                # Manually construct 4x4 matrix: [[R, p], [0, 1]]
                batch_size = pos.shape[0]
                pose_4x4 = torch.eye(4, device=pose_data.device, dtype=pose_data.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
                pose_4x4[:, :3, :3] = rot_mat  # Top-left 3x3: rotation
                pose_4x4[:, :3, 3] = pos  # Top-right 3x1: position
                
                # Restore original dimensionality: if input was 1D, output 2D (4x4); if 2D, output 3D (Nx4x4)
                if original_dim == 1:
                    converted_object_poses[obj_name] = pose_4x4[0]
                else:
                    converted_object_poses[obj_name] = pose_4x4
            else:
                # Already in 4x4 format or other format - pass through
                converted_object_poses[obj_name] = pose_data

        datagen_info = {
            "object_pose": converted_object_poses,
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask start observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask start observations recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, marked_subtask_action_indices

    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return 0

    # get output directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)

    env_cfg.env_name = env_name

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders = MimicRecorderManagerCfg()
    if not args_cli.auto:
        # disable subtask term signals recorder term if in manual mode
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    if not args_cli.auto or (args_cli.auto and not args_cli.annotate_subtask_start_signals):
        # disable subtask start signals recorder term if in manual mode or no need for subtask start annotations
        env_cfg.recorders.record_pre_step_subtask_start_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment from loaded config
    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        # check if the mimic API env.get_subtask_term_signals() is implemented
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
        if (
            args_cli.annotate_subtask_start_signals
            and env.get_subtask_start_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_start_signals
        ):
            raise NotImplementedError(
                "The environment does not implement the get_subtask_start_signals method required "
                "to run automatic annotations."
            )
    else:
        # get subtask termination signal names for each eef from the environment configs
        subtask_term_signal_names = {}
        subtask_start_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_start_signal_names[eef_name] = (
                [subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs]
                if args_cli.annotate_subtask_start_signals
                else []
            )
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            # Validation: if annotating start signals, every subtask (including the last) must have a name
            if args_cli.annotate_subtask_start_signals:
                if any(name in (None, "") for name in subtask_start_signal_names[eef_name]):
                    raise ValueError(
                        f"Missing 'subtask_term_signal' for one or more subtasks in eef '{eef_name}'. When"
                        " '--annotate_subtask_start_signals' is enabled, each subtask (including the last) must"
                        " specify 'subtask_term_signal'. The last subtask's term signal name is used as the final"
                        " start signal name."
                    )
            # no need to annotate the last subtask term signal, so remove it from the list
            subtask_term_signal_names[eef_name].pop()

    # reset environment
    env.reset()

    # Only enables inputs if this script is NOT headless mode
    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()

    # simulate environment -- run everything in inference mode
    exported_episode_count = 0
    processed_episode_count = 0
    successful_task_count = 0  # Counter for successful task completions
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Iterate over the episodes in the loaded dataset file
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                is_episode_annotated_successfully = False
                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                        env, episode, success_term, subtask_term_signal_names, subtask_start_signal_names
                    )

                if is_episode_annotated_successfully and not skip_episode:
                    # set success to the recorded episode data and export to file
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    successful_task_count += 1  # Increment successful task counter
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode due to incomplete subtask annotations.")
            break

    print(
        f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
        f" episode{'s' if exported_episode_count > 1 else ''}."
    )
    print(
        f"Successful task completions: {successful_task_count}"
    )  # This line is used by the dataset generation test case to check if the expected number of demos were annotated
    print("Exiting the app.")

    # Close environment after annotation is complete
    env.close()

    return successful_task_count


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Replays an episode in the environment.

    This function replays the given recorded episode in the environment. It can optionally check if the task
    was successfully completed using a success termination condition input.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully replayed and the success condition was met (if provided),
        False otherwise.
    """
    global current_action_index, skip_episode, is_paused
    # read initial state and actions from the loaded episode
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, None, is_relative=True)
    
    # Set object poses from the recorded initial state if available
    if "rigid_object" in initial_state:
        for obj_name, obj_data in initial_state["rigid_object"].items():
            if "root_pose" in obj_data:
                pose_data = obj_data["root_pose"]
                # Handle both numpy arrays and tensors
                if isinstance(pose_data, torch.Tensor):
                    root_pose = pose_data.float().to(env.device)
                else:
                    root_pose = torch.from_numpy(pose_data).float().to(env.device)
                #print(f"Setting initial pose for object '{obj_name}' to {root_pose.cpu().numpy()}")
                try:
                    env.set_object_pose(root_pose, obj_name=obj_name)
                except (AttributeError, RuntimeError, ValueError) as e:
                    print(f"Warning: Could not set object pose for {obj_name}: {e}")
                    pass
    
    first_action = True
    for action_index, action in enumerate(actions):
        current_action_index = action_index
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue
        action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
        env.step(torch.Tensor(action_tensor))
    if success_term is not None:
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Annotates an episode in automatic mode.

    This function replays the given episode in the environment and checks if the task was successfully completed.
    If the task was not completed, it will print a message and return False. Otherwise, it will check if all the
    subtask term signals are annotated and return True if they are, False otherwise.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global skip_episode
    skip_episode = False
    is_episode_annotated_successfully = replay_episode(env, episode, success_term)
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
    else:
        # check if all the subtask term signals are annotated
        annotated_episode = env.recorder_manager.get_episode(0)
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            signal_flags = torch.tensor(signal_flags, device=env.device)
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')
        if args_cli.annotate_subtask_start_signals:
            subtask_start_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_start_signals"]
            for signal_name, signal_flags in subtask_start_signal_dict.items():
                if not torch.any(signal_flags):
                    is_episode_annotated_successfully = False
                    print(f'\tDid not detect start for the subtask "{signal_name}".')
    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    subtask_start_signal_names: dict[str, list[str]] = {},
) -> bool:
    """Annotates an episode in manual mode.

    This function replays the given episode in the environment and allows for manual marking of subtask term signals.
    It iterates over each eef and prompts the user to mark the subtask term signals for that eef.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.
        subtask_term_signal_names: Dictionary mapping eef names to lists of subtask term signal names.
        subtask_start_signal_names: Dictionary mapping eef names to lists of subtask start signal names.
    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global is_paused, marked_subtask_action_indices, skip_episode
    # iterate over the eefs for marking subtask term signals
    subtask_term_signal_action_indices = {}
    subtask_start_signal_action_indices = {}
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        eef_subtask_start_signal_names = subtask_start_signal_names[eef_name]
        # skip if no subtask annotation is needed for this eef
        if len(eef_subtask_term_signal_names) == 0 and len(eef_subtask_start_signal_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            if len(eef_subtask_start_signal_names) > 0:
                print(f"\t\t- Start:\t{eef_subtask_start_signal_names}")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")

            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            task_success_result = replay_episode(env, episode, success_term)
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_subtask_signal_count = len(eef_subtask_term_signal_names) + len(eef_subtask_start_signal_names)
            if task_success_result and expected_subtask_signal_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                for marked_signal_index in range(expected_subtask_signal_count):
                    if args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 0:
                        subtask_start_signal_action_indices[
                            eef_subtask_start_signal_names[int(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                    if not args_cli.annotate_subtask_start_signals:
                        # Direct mapping when only collecting termination signals
                        subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                            marked_subtask_action_indices[marked_signal_index]
                        )
                    elif args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 1:
                        # Every other signal is a termination when collecting both types
                        subtask_term_signal_action_indices[
                            eef_subtask_term_signal_names[math.floor(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                break

            if not task_success_result:
                print("\tThe final task was not completed.")
                return False

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    annotated_episode = env.recorder_manager.get_episode(0)
    for (
        subtask_term_signal_name,
        subtask_term_signal_action_index,
    ) in subtask_term_signal_action_indices.items():
        # subtask termination signal is false until subtask is complete, and true afterwards
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)

    if args_cli.annotate_subtask_start_signals:
        for (
            subtask_start_signal_name,
            subtask_start_signal_action_index,
        ) in subtask_start_signal_action_indices.items():
            subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
            subtask_signals[:subtask_start_signal_action_index] = False
            annotated_episode.add(
                f"obs/datagen_info/subtask_start_signals/{subtask_start_signal_name}", subtask_signals
            )

    return True


if __name__ == "__main__":
    # run the main function
    successful_task_count = main()
    # close sim app
    simulation_app.close()
    # exit with the number of successful task completions as return code
    exit(successful_task_count)
