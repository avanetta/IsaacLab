import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


FR3_WITH_HAND_CFG = ArticulationCfg(
    # Load USD
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/pdz/franka_ros2_ws/src/franka_description/robots/fr3/fr3_arm/fr3_arm.usd",
        # fix_root_link removed – FR3 USD already has correct root
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0, fix_root_link=True
        ),
    ),

    # Initial joint state
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0444,
            "fr3_joint2": -0.1894,
            "fr3_joint3": -0.1107,
            "fr3_joint4": -2.5148,
            "fr3_joint5":  0.044,
            "fr3_joint6": 3.3775,
            "fr3_joint7": 0.6952,
            "fr3_finger_joint1": 0.04,
            "fr3_finger_joint2": 0.04,
        }
    ),

    # Actuation model
    actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[1-7]"],
        effort_limit_sim=87.0,  # Changed from effort_limit
        stiffness=400.0,
        damping=40.0,
    ),
    "fr3_hand": ImplicitActuatorCfg(
        joint_names_expr=["fr3_finger_joint.*"],
        effort_limit_sim=200.0,  # Changed from effort_limit
        stiffness=2000.0,
        damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,


)
