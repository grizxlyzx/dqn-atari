import numpy as np

class FIFOReplayBuffer:
    def __init__(self, capacity, *buffers):
        self._capacity = capacity
        self._buffer = np.zeros([len(buffers), capacity])



    def _init_add(self, *args):
        pass

    def _add(self):
        pass

    def sample(self, batch_size):
        pass

    def size(self):
        return self.__len__()

    def __len__(self):
        pass

    def clear(self):
        pass

