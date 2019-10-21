import numpy as np

from simulator.utils import unimplemented


class Interrupt(object):

    def tick(self):
        unimplemented()

    def spin(self, env):
        while not self.tick():
            env.step()


class NumericStateInterrupt(Interrupt):

    def __init__(self, target_state, state_reader, tolerance=1e-4):
        super(NumericStateInterrupt, self).__init__()
        self._state_reader = state_reader
        self._target_state = target_state
        self._tolerance = tolerance

    def tick(self):
        current_state = self._state_reader()
        return np.all(np.abs(current_state - self._target_state) <= self._tolerance)


class ComposeInterrupts(Interrupt):

    def __init__(self, interrupts, wait_for_all=True):
        super(ComposeInterrupts, self).__init__()
        self._interrupts = interrupts
        self._interrupted = []
        self._wait_for_all = wait_for_all

    def tick(self):
        self._interrupted = [i for i in self._interrupts if i.tick()]

        count = len(self._interrupts) if self._wait_for_all else 1
        return len(self._interrupted) >= count


def compose_interrupts(*interrupts):
    return ComposeInterrupts(interrupts)
