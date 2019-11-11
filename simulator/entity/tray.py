import os
import numpy as np

import pybullet as pb
import pybullet_data

from .base import Entity
from .cube import Cube


class Tray(Entity):

    def __init__(self, pb_client=pb, position=(0, 0, 0), orientation=(0, 0, 0, 1), fix_base=False,
                 scale=1., color=None, **kwargs):
        table = os.path.join(pybullet_data.getDataPath(), 'tray', 'traybox.urdf')
        super(Tray, self).__init__(table, pb_client, position, orientation, fix_base, scale, **kwargs)

        if color is not None:
            self._pb_client.changeVisualShape(self._id, -1, rgbaColor=color)

    def add_cube(self, position, orientation):
        position, orientation = self.transform(position, orientation)
        return Cube(pb_client=self._pb_client, position=position, orientation=orientation, debug=self._debug)

    def add_random_cube(self):
        bb = self.bounding_box
        pos = np.random.uniform(bb[0] + [0.025, 0.025, self.z_end], bb[1] - [0.025, 0.025, self.z_end+0.1])
        ori = np.random.uniform((0, ) * 3, (np.pi * 2, ) * 3)
        cube = Cube(pb_client=self._pb_client, position=pos,
                    orientation=self._pb_client.getQuaternionFromEuler(ori), debug=self._debug)
        return cube
