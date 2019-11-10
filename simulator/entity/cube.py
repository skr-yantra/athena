import os

import pybullet as pb
import pybullet_data

from .base import Entity


class Cube(Entity):

    def __init__(self, pb_client=pb, position=(0, 0, 0), orientation=(0, 0, 0, 1), fix_base=False, scale=0.5, **kwargs):
        table = os.path.join(pybullet_data.getDataPath(), 'cube_small.urdf')
        super(Cube, self).__init__(table, pb_client, position, orientation, fix_base, scale, **kwargs)
