import math

from .base import Environment
from ..entity.ground import Ground
from ..entity.irb120 import IRB120
from ..entity.table import Table
from ..entity.tray import Tray
from .. import interrupts


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

    def act(self, dposition, dorientation, gripper_opened):
        pose_interrupt = self._robot.move_gripper_pose(dposition, self._pb_client.getQuaternionFromEuler(dorientation))
        gripper_interrupt = self._robot.set_gripper_finger(gripper_opened)

        return interrupts.all(pose_interrupt, gripper_interrupt)
