import os

import pybullet as pb
import pybullet_data

from .base import Entity


class Table(Entity):

    def __init__(self, pb_client=pb, position=(0, 0, 0), orientation=(0, 0, 0, 1), fix_base=True, scale=1., **kwargs):
        table = os.path.join(pybullet_data.getDataPath(), 'table', 'table.urdf')
        super(Table, self).__init__(table, pb_client, position, orientation, fix_base, scale, **kwargs)
