import gymnasium as gym

# List all Isaac Lab unplug environments
all_envs = [env_id for env_id in gym.envs.registry.keys()]
print("Available Unplug Environments:")
for env_id in all_envs:
    print(f"  - {env_id}")