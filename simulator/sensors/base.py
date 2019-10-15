import pybullet as pb

from ..utils import unimplemented


class Sensor(object):

    def __init__(self, pb_client=pb):
        self._pb_client = pb_client

    @property
    def state(self):
        unimplemented()
