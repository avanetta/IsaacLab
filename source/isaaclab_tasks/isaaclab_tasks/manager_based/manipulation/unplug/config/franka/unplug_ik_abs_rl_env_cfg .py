from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Import your base configuration
from .unplug_ik_abs_env_cfg import FrankaUnplugIKAbsEnvCfgRGB
from isaaclab_tasks.manager_based.manipulation.unplug import mdp

@configclass
class FrankaUnplugIKAbsRLEnvCfgRGB(FrankaUnplugIKAbsEnvCfgRGB):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 1024 
        self.rewards = RewardsCfg()
        self.observations.policy.concatenate_terms = True
        
        # Use your existing termination function for the 'success' exit condition
        self.terminations.success = DoneTerm(
            func=mdp.plug_successfully_unplugged,
            params={
                "unplug_distance": 0.15,
            }
        )

@configclass
class RewardsCfg:
    """Reward terms for the RL agent."""

    # -- Reach: Standard shaping reward
    reaching_plug = RewTerm(
        func=mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"), 
            "target_cfg": SceneEntityCfg("plug")
        }
    )

    # -- Grasp: Using your function from terminations.py
    is_grasped = RewTerm(
        func=mdp.is_plug_grasped,
        weight=5.0,
        params={
            "gripper_open_val": 0.04,
            "dist_threshold": 0.05
        }
    )

    # -- Success: Using your function from terminations.py
    unplug_success = RewTerm(
        func=mdp.plug_successfully_unplugged,
        weight=50.0,
        params={
            "unplug_distance": 0.15,
        }
    )

    # -- Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)