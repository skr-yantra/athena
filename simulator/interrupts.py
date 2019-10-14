import numpy as np

from simulator.utils import unimplemented


class Interrupt(object):

    def __init__(self, env):
        self._env = env

    def interrupt_tick(self):
        unimplemented()


class NumericStateInterrupt(Interrupt):

    def __init__(self, env, target_state, state_reader, tolerance=1e-4):
        super(NumericStateInterrupt, self).__init__(env)
        self._state_reader = state_reader
        self._target_state = target_state
        self._tolerance = tolerance

    def interrupt_tick(self):
        current_state = self._state_reader(self._env)
        return np.all(np.abs(current_state - self._target_state) <= self._tolerance)
