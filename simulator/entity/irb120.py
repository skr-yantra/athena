import pybullet as pb
import numpy as np

from ..data import abb_irb120
from .base import Entity
from ..sensors.camera import Camera
from ..interrupts import NumericStateInterrupt

REVOLUTE_JOINT_INDICES = (1, 2, 3, 4, 5, 6)
GRIPPER_INDEX = 7
GRIPPER_FINGER_INDICES = (8, 9)
MOVABLE_JOINT_INDICES = REVOLUTE_JOINT_INDICES + GRIPPER_FINGER_INDICES
FINGER_JOINT_RANGE = np.array([
    [0., 0.012],
    [0, 0.012],
])


class IRB120(Entity):

    def __init__(self, pb_client=pb, position=(0, 0, 0), orientation=(0, 0, 0, 1), fixed=True, scale=1., max_finger_force=200.):
        urdf = abb_irb120()
        super(IRB120, self).__init__(urdf, pb_client, position, orientation, fixed, scale)

        self._max_finger_force = max_finger_force

        self._gripper_cam = Camera(
            self._pb_client,
            near_plane=0.001,
            far_plane=1.,
            view_calculator=self._gripper_cam_view_calculator,
            debug=False
        )

    @property
    def revolute_joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._id, REVOLUTE_JOINT_INDICES)])

    @property
    def finger_joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._id, GRIPPER_FINGER_INDICES)])

    @property
    def gripper_pose_euler(self):
        gripper_state = self._pb_client.getLinkState(self._id, GRIPPER_INDEX)
        px, py, pz = gripper_state[0]
        ax, ay, az = pb.getEulerFromQuaternion(gripper_state[1])

        return np.array([px, py, pz]), np.array([ax, ay, az])

    @property
    def gripper_pose(self):
        gripper_state = self._pb_client.getLinkState(self._id, GRIPPER_INDEX)
        return np.array(gripper_state[0]), np.array(gripper_state[1])

    def set_gripper_pose(self, position, orientation):
        joint_states = pb.calculateInverseKinematics(
            self._id,
            GRIPPER_INDEX,
            position,
            orientation
        )[:-2]

        return self.set_revolute_joint_state(joint_states)

    def move_gripper_pose(self, dposition, dorientation):
        current_position, current_orientation = self.gripper_pose

        _, orientation = self._pb_client.multiplyTransforms(
            (0, 0, 0),
            current_orientation,
            (0, 0, 0),
            dorientation
        )

        return self.set_gripper_pose(
            current_position + dposition,
            orientation
        )

    def set_revolute_joint_state(self, joint_states):
        assert len(REVOLUTE_JOINT_INDICES) == len(joint_states)

        self._pb_client.setJointMotorControlArray(
            self._id,
            REVOLUTE_JOINT_INDICES,
            pb.POSITION_CONTROL,
            joint_states
        )

        return self._make_revolute_joint_interrupt(joint_states)

    def set_gripper_finger(self, open):
        if open:
            return self.open_gripper()
        else:
            return self.close_gripper()

    def open_gripper(self):
        return self.set_finger_joint_state(FINGER_JOINT_RANGE[:, 1].ravel())

    def close_gripper(self):
        return self.set_finger_joint_state(FINGER_JOINT_RANGE[:, 0].ravel())

    def set_finger_joint_state(self, joint_states):
        assert len(joint_states) == len(GRIPPER_FINGER_INDICES)

        self._pb_client.setJointMotorControlArray(
            self._id,
            GRIPPER_FINGER_INDICES,
            pb.POSITION_CONTROL,
            joint_states,
            forces=(self._max_finger_force,) * 2
        )

        return self._make_finger_joint_interrupt(joint_states)

    def _make_revolute_joint_interrupt(self, target_state):
        assert len(target_state) == len(REVOLUTE_JOINT_INDICES)
        interrupt = NumericStateInterrupt(target_state, lambda: self.revolute_joint_state)
        return interrupt

    def _make_finger_joint_interrupt(self, target_state):
        assert len(target_state) == len(GRIPPER_FINGER_INDICES)
        interrupt = NumericStateInterrupt(target_state, lambda: self.finger_joint_state)
        return interrupt

    def capture_gripper_camera(self):
        return self._gripper_cam.state

    def _gripper_cam_view_calculator(self):
        position, orientation = self.gripper_pose

        eye, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (0, 0, 0.05),
            (1, 0, 0, 0)
        )

        to, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (0.1, 0, 0.05),
            (1, 0, 0, 0)
        )

        up, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (0, 0, 0.15),
            (1, 0, 0, 0)
        )

        return eye, to, up
