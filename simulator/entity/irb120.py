import pybullet as pb
import numpy as np

from ..data import abb_irb120
from .base import Entity

REVOLUTE_JOINT_INDICES = (1, 2, 3, 4, 5, 6)
GRIPPER_INDEX = 7
GRIPPER_FINGER_INDICES = (8, 9)
MOVABLE_JOINT_INDICES = REVOLUTE_JOINT_INDICES + GRIPPER_FINGER_INDICES
FINGER_JOINT_RANGE = np.array([
    [0., 0.012],
    [0, 0.012],
])


class IRB120(Entity):

    def __init__(self, pb_client=pb, pose=(0., 0., 0., 0., 0., 0.), fixed=True, scale=1., max_finger_force=200.):
        urdf = abb_irb120()
        super(IRB120, self).__init__(urdf, pb_client, pose, fixed, scale)

        self._max_finger_force = max_finger_force

    @property
    def revolute_joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._id, REVOLUTE_JOINT_INDICES)])

    @property
    def finger_joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._id, GRIPPER_FINGER_INDICES)])

    @property
    def gripper_pose(self):
        gripper_state = self._pb_client.getLinkState(self._id, GRIPPER_INDEX)
        px, py, pz = gripper_state[0]
        ax, ay, az = pb.getEulerFromQuaternion(gripper_state[1])

        return np.array([px, py, pz, ax, ay, az])

    @property
    def gripper_pose_quaternion(self):
        gripper_state = self._pb_client.getLinkState(self._id, GRIPPER_INDEX)
        return gripper_state[0], gripper_state[1]

    def set_gripper_pose(self, pose):
        joint_states = pb.calculateInverseKinematics(
            self._id,
            GRIPPER_INDEX,
            pose[:3],
            pb.getQuaternionFromEuler(pose[3:])
        )[:-2]

        self.set_revolute_joint_state(joint_states)

    def move_gripper_pose(self, dpose):
        return self.set_gripper_pose(self.gripper_pose + dpose)

    def set_revolute_joint_state(self, joint_states):
        assert len(REVOLUTE_JOINT_INDICES) == len(joint_states)

        self._pb_client.setJointMotorControlArray(
            self._id,
            REVOLUTE_JOINT_INDICES,
            pb.POSITION_CONTROL,
            joint_states
        )

    def open_gripper(self):
        self.set_finger_joint_state(FINGER_JOINT_RANGE[:, 1].ravel())

    def close_gripper(self):
        self.set_finger_joint_state(FINGER_JOINT_RANGE[:, 0].ravel())

    def set_finger_joint_state(self, joint_states):
        assert len(joint_states) == len(GRIPPER_FINGER_INDICES)

        self._pb_client.setJointMotorControlArray(
            self._id,
            GRIPPER_FINGER_INDICES,
            pb.POSITION_CONTROL,
            joint_states,
            forces=(self._max_finger_force,) * 2
        )
