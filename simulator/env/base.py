import time

import pybullet as pb

from ..utils import unimplemented


class Environment(object):

    def __init__(self, pb_client=pb):
        self._pb_client = pb_client
        self._setup()

    def spin(self):
        while True:
            self.step()
            time.sleep(1./240.)

    def step(self):
        self._pb_client.stepSimulation()

    def _setup(self):
        self._step = 0

    def reset(self):
        self._pb_client.resetSimulation()
        self._setup()
