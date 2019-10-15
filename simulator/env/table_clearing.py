import math

import pybullet as pb

from ..entity.irb120 import IRB120
from ..entity.ground import Ground
from ..entity.table import Table
from ..entity.tray import Tray
from ..sensors.camera import Camera
from .base import Environment


class TableClearingEnvironment(Environment):

    def _setup(self):
        super(TableClearingEnvironment, self)._setup()

        self._pb_client.setGravity(0, 0, -9.81)

        self._ground = Ground(self._pb_client)
        self._robot = IRB120(self._pb_client)
        self._src_table = Table(self._pb_client, position=(0, -0.5, 0), scale=0.5)
        self._dest_table = Table(self._pb_client, position=(0, 0.5, 0), scale=0.5)

        self._src_tray = Tray(self._pb_client, position=(0, -0.5, self._src_table.z_end), scale=0.5)
        self._dest_tray = Tray(self._pb_client, position=(0, 0.5, self._dest_table.z_end), scale=0.5)

        self._gripper_cam = Camera(
            self._pb_client,
            near_plane=0.001,
            far_plane=1.,
            pose_reader=lambda: self._robot.gripper_pose,
            debug=False
        )

        for i in range(5):
            self._src_tray.add_random_cube()

        self._robot.set_gripper_pose((0, -0.7, 0.5), self._pb_client.getQuaternionFromEuler((math.pi/2, math.pi/2, 0)))
        self._robot.open_gripper()

    def step(self):
        super(TableClearingEnvironment, self).step()

        if self._step % (240/10) == 0:
            self._gripper_cam.state

