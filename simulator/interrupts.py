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

    def __init__(self, interrupts, decision_maker):
        super(ComposeInterrupts, self).__init__()
        self._interrupts = interrupts
        self._interrupted = []
        self._decision_maker = decision_maker

    @property
    def interrupts(self):
        return self._interrupts

    def tick(self):
        self._interrupted = [i for i in self._interrupts if i.tick()]
        return self._decision_maker(self, self._interrupted)


def all(*interrupts):
    return ComposeInterrupts(interrupts, lambda i, v: len(i.interrupts) == len(v))


def any(*interrupts):
    return ComposeInterrupts(interrupts, lambda _, v: len(v) > 0)


def compose(*interrupts, decision_maker):
    return ComposeInterrupts(interrupts, decision_maker)
