import collections
import numpy as np
import torch

from training.buffer import Experience


class Agent:
    def __init__(self, env, buffer, n_steps=1, gamma=0.99):
        self.env = env
        self.buffer = buffer
        self.n_steps = n_steps
        self.gamma = gamma
        self._nstep_queue = collections.deque(maxlen=n_steps)
        self._reset()

    def _reset(self, **kwargs):
        self.state = self.env.reset(**kwargs)[0]
        self.total_reward = 0.0

    def _make_nstep(self):
        s, a, _, _, _ = self._nstep_queue[0]
        n_reward = 0.0
        n_next_state = None
        n_done = False
        for i, (_, _, r, ns, d) in enumerate(self._nstep_queue):
            n_reward += (self.gamma ** i) * r
            n_next_state = ns
            if d:
                n_done = True
                break
        return Experience(s, a, n_reward, n_next_state, n_done)

    @torch.inference_mode()
    def play_step(self, net, epsilon=0.0, device="cuda"):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.from_numpy(state_a).to(device).float() / 255.0
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        self._nstep_queue.append((self.state, action, reward, new_state, done))

        if done:
            while self._nstep_queue:
                self.buffer.append(self._make_nstep())
                self._nstep_queue.popleft()
            done_reward = self.total_reward
            self._reset()
        else:
            if len(self._nstep_queue) == self.n_steps:
                self.buffer.append(self._make_nstep())
            self.state = new_state

        return done_reward
