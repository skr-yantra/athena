import os

import pybullet as pb
import pybullet_data

from .base import Entity


class Ground(Entity):

    def __init__(self, pb_client=pb):
        plane = os.path.join(pybullet_data.getDataPath(), 'plane.urdf')
        super(Ground, self).__init__(plane, pb_client)
