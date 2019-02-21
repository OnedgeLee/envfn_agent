import numpy as np

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        for row in range(s.shape[0]):
            transition = np.hstack((s[row], a[row], r[row], s_[row]))
            index = self.pointer % self.capacity
            self.data[index, :] = transition
            self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]