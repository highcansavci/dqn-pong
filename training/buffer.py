import collections
import numpy as np

Experience = collections.namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def clear(self):
        self.buffer.clear()


class SumTree:
    """Binary segment tree storing priorities at leaves and partial sums at internal nodes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedExperienceBuffer:
    """Proportional PER (Schaul et al. 2016) backed by a sum tree."""

    def __init__(self, capacity, alpha=0.6, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0  # untransformed |TD| upper bound seen so far

    def __len__(self):
        return self.tree.n_entries

    def append(self, experience):
        # New transitions enter at max priority so they're sampled at least once
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size, beta=0.4):
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        experiences = []
        total = self.tree.total()
        segment = total / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = p
            experiences.append(data)

        sampling_probs = priorities / total
        weights = (len(self) * sampling_probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        abs_td = np.abs(td_errors) + self.epsilon
        for idx, td in zip(indices, abs_td):
            self.tree.update(idx, td ** self.alpha)
            if td > self.max_priority:
                self.max_priority = td
