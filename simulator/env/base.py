import time

import pybullet as pb

from ..utils import unimplemented


class Environment(object):

    def __init__(self, pb_client=pb):
        self._pb_client = pb_client
        self._setup()

    def spin(self):
        while True:
            pb.stepSimulation()
            time.sleep(1./240.)

    def _setup(self):
        unimplemented()

    def reset(self):
        self._pb_client.resetSimulation()
        self._setup()
