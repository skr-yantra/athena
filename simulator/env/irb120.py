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

DEFAULT_POSITION_TOLERANCE = 1e-6
DEFAULT_ORIENTATION_TOLERANCE = 1e-2
DEFAULT_TOLERANCE = (DEFAULT_POSITION_TOLERANCE, ) * 3 + (DEFAULT_ORIENTATION_TOLERANCE, ) * 3


class IRB120(object):

    def __init__(self, pb_client=pb, gravity=(0, 0, -9.81), realtime=True, pose_tolerance=DEFAULT_POSITION_TOLERANCE):
        self._urdf_robot = abb_irb120()
        assert_exist(self._urdf_robot)

        self._urdf_plane = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
        assert_exist(self._urdf_plane)

        assert pb_client is not None
        self._pb_client = pb_client

        self._gravity = gravity
        self._realtime = realtime
        self._pose_tolerance = np.array(pose_tolerance)

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

    def reset(self):
        pass

    def move_absolute(self, pose):
        px, py, pz, ax, ay, az = pose

        joint_states = pb.calculateInverseKinematics(
            self._robot_id,
            GRIPPER_INDEX,
            (px, py, pz),
            pb.getQuaternionFromEuler((ax, ay, az)),
            maxNumIterations=100,
            residualThreshold=1e-6
        )

        self._move_joint(joint_states)
        self._wait_for_gripper_pose(pose)

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

    def _wait_for_gripper_pose(self, target_pose):
        while np.any(np.abs(np.array(target_pose) - self.gripper_pose) > self._pose_tolerance):
            self._tick()
            print(np.abs(np.array(target_pose) - self.gripper_pose))

    def _tick(self):
        if self._realtime:
            time.sleep(1./240.)

        self._pb_client.stepSimulation()

