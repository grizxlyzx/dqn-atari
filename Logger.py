import numpy as np

class Logger:
    __slots__ = ('_log', '_epi_log')

    def __init__(self):
        self._log = []
        self._epi_log = []

    def log(self, val):
        self._epi_log.append(val)

    def new_epoch(self):
        if self._epi_log:
            self._log.append(np.array(self._epi_log, dtype=np.float32))
            self._epi_log = []

    def latest_mean(self, num_epi):
        avg = 0
        num_epi = min(len(self._log), num_epi)
        for i in range(num_epi):
            avg += self._log[-(i + 1)].mean()
        return avg / num_epi if num_epi > 0 else avg

    def latest_sum(self, num_epi):
        sum = 0
        num_epi = min(len(self._log), num_epi)
        for i in range(num_epi):
            sum += self._log[-(i + 1)].sum()
        return sum / num_epi if num_epi > 0 else sum

    def save(self, fn):
        pass

