import os

import pybullet as pb
import pybullet_data

from .base import Entity


class Cube(Entity):

    def __init__(self, pb_client=pb, pose=(0, 0, 0, 0, 0, 0), fix_base=False, scale=0.5):
        table = os.path.join(pybullet_data.getDataPath(), 'cube_small.urdf')
        super(Cube, self).__init__(table, pb_client, pose, fix_base, scale)
