import pybullet as pb
import numpy as np

from simulator.utils import unimplemented


class Interrupt(object):

    def should_interrupt(self):
        unimplemented()

    def spin(self, env):
        while not self.should_interrupt():
            env.step()


class NumericStateInterrupt(Interrupt):

    def __init__(self, target_state, state_reader, tolerance=1e-4):
        super(NumericStateInterrupt, self).__init__()
        self._state_reader = state_reader
        self._target_state = target_state
        self._tolerance = tolerance

    def should_interrupt(self):
        current_state = self._state_reader()
        return np.all(np.abs(current_state - self._target_state) <= self._tolerance)


class CollisionInterrupt(Interrupt):

    def __init__(self, target, exclusions=tuple(), pb_client=pb):
        super(CollisionInterrupt, self).__init__()
        self._pb_client = pb_client
        self._target = target
        self._exclusions = exclusions

    def should_interrupt(self):
        points = self._pb_client.getContactPoints(self._target)
        collisions = [p[2] for p in points if p[2] not in self._exclusions]
        return len(collisions) > 0


class ComposeInterrupts(Interrupt):

    def __init__(self, interrupts, decision_maker):
        super(ComposeInterrupts, self).__init__()
        self._interrupts = interrupts
        self._interrupted = []
        self._decision_maker = decision_maker

    @property
    def interrupts(self):
        return self._interrupts

    def should_interrupt(self):
        self._interrupted = [i for i in self._interrupts if i.should_interrupt()]
        return self._decision_maker(self, self._interrupted)


def all(*interrupts):
    return ComposeInterrupts(interrupts, lambda i, v: len(i.interrupts) == len(v))


def any(*interrupts):
    return ComposeInterrupts(interrupts, lambda _, v: len(v) > 0)


def compose(*interrupts, decision_maker):
    return ComposeInterrupts(interrupts, decision_maker)
