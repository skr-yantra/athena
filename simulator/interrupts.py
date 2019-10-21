import numpy as np

from simulator.utils import unimplemented


class Interrupt(object):

    def tick(self):
        unimplemented()


class NumericStateInterrupt(Interrupt):

    def __init__(self, target_state, state_reader, tolerance=1e-4):
        super(NumericStateInterrupt, self).__init__()
        self._state_reader = state_reader
        self._target_state = target_state
        self._tolerance = tolerance

    def tick(self):
        current_state = self._state_reader()
        return np.all(np.abs(current_state - self._target_state) <= self._tolerance)
