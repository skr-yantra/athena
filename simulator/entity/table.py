import os

import pybullet as pb
import pybullet_data

from .base import Entity


class Table(Entity):

    def __init__(self, pb_client=pb, pose=(0, 0, 0, 0, 0, 0), fix_base=True, scale=1.):
        table = os.path.join(pybullet_data.getDataPath(), 'table', 'table.urdf')
        super(Table, self).__init__(table, pb_client, pose, fix_base, scale)
