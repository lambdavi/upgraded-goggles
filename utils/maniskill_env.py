import gymnasium as gym
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper

def make_maniskill_env(env_id, obs_mode, control_mode, num_envs):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        num_envs=num_envs,
        render_mode="rgb_array"  # required for camera access
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env
