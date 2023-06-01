import numpy as np

class CircularReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._buffer = np.empty(capacity, dtype=object)
        self._write_ptr = 0
        self._size = 0

    def add(self, ob, ob_nx, action, reward, reward_nx, done):
        self._buffer[self._write_ptr] = [ob, ob_nx, action, reward, reward_nx, done]
        self._size += 1 if self._size < self._capacity else 0
        self._write_ptr += 1
        self._write_ptr %= self._capacity

    def sample(self, batch_size):
        transitions = np.random.choice(self._buffer[:self._size], batch_size, replace=False)
        ob, ob_nx, action, reward, reward_nx, done = zip(*transitions)
        return np.array(ob), np.array(ob_nx), action, reward, reward_nx, done

    def size(self):
        return self.__len__()

    def __len__(self):
        return self._size

    def clear(self):
        self._buffer = np.empty(self._capacity, dtype=object)
        self._write_ptr = 0
        self._size = 0

