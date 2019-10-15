import time

import pybullet as pb

from ..utils import unimplemented


class Environment(object):

    def __init__(self, pb_client=pb, step_size=1./240.):
        self._pb_client = pb_client

        self._step_size = step_size
        self._last_step_time = time.time_ns()

        self._setup()

    def spin(self):
        while True:
            self.step()

    def step(self):
        self._pb_client.stepSimulation()
        current = time.time_ns()
        elapsed = current - self._last_step_time
        if elapsed < self._step_size:
            time.sleep(self._step_size-elapsed)

    def _setup(self):
        self._step = 0

    def reset(self):
        self._pb_client.resetSimulation()
        self._setup()
