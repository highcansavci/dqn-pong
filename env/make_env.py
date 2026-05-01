import gymnasium as gym

from wrappers.max_and_skip_env import MaxAndSkipEnv
from wrappers.warp_frame import WarpFrame
from wrappers.buffer_wrapper import BufferWrapper

import ale_py

gym.register_envs(ale_py)


def make_env(env_id, skip=4, n_steps=4):
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=skip)
    env = WarpFrame(env)
    env = BufferWrapper(env, n_steps=n_steps)
    return env
