import os
import time

import numpy as np
import pybullet as pb
import pybullet_data

from ..data import abb_irb120
from ..utils import assert_exist


REVOLUTE_JOINT_INDICES = (1, 2, 3, 4, 5, 6)
GRIPPER_INDEX = 7
GRIPPER_FINGER_INDICES = (8, 9)
MOVABLE_JOINT_INDICES = REVOLUTE_JOINT_INDICES + GRIPPER_FINGER_INDICES


class IRB120(object):

    def __init__(self, pb_client=pb, gravity=(0, 0, -9.81), realtime=True, joint_state_tolerance=1e-4):
        self._urdf_robot = abb_irb120()
        assert_exist(self._urdf_robot)

        self._urdf_plane = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
        assert_exist(self._urdf_plane)

        assert pb_client is not None
        self._pb_client = pb_client

        self._gravity = gravity
        self._realtime = realtime
        self._joint_state_tolerance = joint_state_tolerance

        self._plane_id = None
        self._robot_id = None

    def setup(self):
        is_connected, _ = self._pb_client.getConnectionInfo()
        if not is_connected:
            raise Exception('Bullet physics client not connected')

        gx, gy, gz = self._gravity
        self._pb_client.setGravity(gx, gy, gz)

        self._plane_id = self._pb_client.loadURDF(self._urdf_plane)
        self._robot_id = self._pb_client.loadURDF(self._urdf_robot, useFixedBase=True)

    @property
    def gripper_pose(self):
        gripper_state = self._pb_client.getLinkState(self._robot_id, GRIPPER_INDEX)
        px, py, pz = gripper_state[0]
        ax, ay, az = pb.getEulerFromQuaternion(gripper_state[1])

        return np.array([px, py, pz, ax, ay, az])

    @property
    def joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._robot_id, MOVABLE_JOINT_INDICES)])

    def reset(self):
        self._pb_client.resetSimulation()
        self.setup()

    def move_absolute(self, pose):
        px, py, pz, ax, ay, az = pose

        joint_states = pb.calculateInverseKinematics(
            self._robot_id,
            GRIPPER_INDEX,
            (px, py, pz),
            pb.getQuaternionFromEuler((ax, ay, az))
        )

        self._move_joint(joint_states)

    def move_relative(self, pose_diff):
        dpx, dpy, dpz, dax, day, daz = pose_diff
        px, py, pz, ax, ay, az = self.gripper_pose

        return self.move_absolute((
            px + dpx, py + dpy, pz + dpz,
            ax + dax, ay + day, az + daz,
        ))

    def move_gripper(self, width):
        pass

    def _move_joint(self, joint_states):
        assert len(MOVABLE_JOINT_INDICES) == len(joint_states)

        self._pb_client.setJointMotorControlArray(
            self._robot_id,
            MOVABLE_JOINT_INDICES,
            pb.POSITION_CONTROL,
            joint_states
        )

        self._wait_for_joint_state(joint_states)

    def _wait_for_joint_state(self, target_state):
        while np.any(np.abs(self.joint_state - target_state) > self._joint_state_tolerance):
            self._tick()

    def _tick(self):
        if self._realtime:
            time.sleep(1./240.)

        self._pb_client.stepSimulation()

