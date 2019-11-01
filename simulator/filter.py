import numpy as np


class MovingAverage(object):

    def __init__(self, count=240, shape=(1, )):
        self._count = count
        self._index = -1
        self._data = np.zeros((count, ) + shape)

    def update(self, value):
        self._index = self._index + 1

        if self._index >= self._count:
            self._index = 0

        self._data[self._index] = value

        return np.mean(self._data, axis=0)
