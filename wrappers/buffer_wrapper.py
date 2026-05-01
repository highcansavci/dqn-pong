import gymnasium as gym
import numpy as np
import gymnasium.spaces as spaces


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.uint8):
        super().__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps

        old_space = env.observation_space  # (H, W, C)
        h, w, c = old_space.shape
        self._channels_per_frame = c

        # Channel-first stack: (C * n_steps, H, W)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c * n_steps, h, w),
            dtype=dtype,
        )
        self.buffer = np.zeros((c * n_steps, h, w), dtype=dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer[:] = 0
        return self.observation(obs), info

    def observation(self, obs):
        # obs: (H, W, C) -> (C, H, W)
        obs = np.transpose(obs, (2, 0, 1))
        c = self._channels_per_frame
        self.buffer[:-c] = self.buffer[c:]
        self.buffer[-c:] = obs
        return self.buffer.copy()
