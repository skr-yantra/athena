import math

import pybullet as pb

from ..entity.irb120 import IRB120
from ..entity.ground import Ground
from ..entity.table import Table
from ..entity.tray import Tray
from .base import Environment


class TableClearingEnvironment(Environment):

    def _setup(self):
        pb.setGravity(0, 0, -9.81)

        self._ground = Ground()
        self._robot = IRB120()
        self._src_table = Table(pose=(0, -0.5, 0, 0, 0, 0), scale=0.5)
        self._dest_table = Table(pose=(0, 0.5, 0, 0, 0, 0), scale=0.5)

        self._src_tray = Tray(pose=(0, -0.5, self._src_table.z_end, 0, 0, 0), scale=0.5)
        self._dest_tray = Tray(pose=(0, 0.5, self._dest_table.z_end, 0, 0, 0), scale=0.5)

        for i in range(50):
            self._src_tray.add_random_cube()

