import math
import logging
from functools import lru_cache

import pybullet as pb
import numpy as np

from .. import interrupts
from ..data import abb_irb120
from .base import Entity
from ..sensors.camera import Camera
from ..interrupts import NumericStateInterrupt, BooleanStateInterrupt
from ..filter import MovingAverage

REVOLUTE_JOINT_INDICES = np.array((1, 2, 3, 4, 5, 6))
GRIPPER_INDEX = 7
GRIPPER_FINGER_INDICES = np.array((8, 9))
MOVABLE_JOINT_INDICES = np.hstack([REVOLUTE_JOINT_INDICES, GRIPPER_FINGER_INDICES])
FINGER_JOINT_RANGE = np.array([
    [0., 0.012],
    [0, 0.012],
])

GRIPPER_ORIGIN_OFFSET = 0.057


def to_deg(ori):
    return np.array(pb.getEulerFromQuaternion(ori)) * 180. / math.pi


class IRB120(Entity):

    def __init__(self, pb_client=pb, position=(0, 0, 0), orientation=(0, 0, 0, 1),
                 fixed=True, scale=1., max_finger_force=200., debug=False, gravity=9.81):
        self._debug = debug

        urdf = abb_irb120()
        super(IRB120, self).__init__(urdf, pb_client, position, orientation, fixed, scale)

        self._max_finger_force = max_finger_force
        self._gravity = gravity

        self._gripper_cam = Camera(
            self._pb_client,
            resolution=(84, 84),
            fov=90.,
            near_plane=0.001,
            far_plane=2.,
            view_calculator=self._gripper_cam_view_calculator,
            debug=debug
        )

        self._pb_client.enableJointForceTorqueSensor(self.id, GRIPPER_FINGER_INDICES[0])
        self._pb_client.enableJointForceTorqueSensor(self.id, GRIPPER_FINGER_INDICES[1])

        self._grasp_interrupt = BooleanStateInterrupt(lambda: self.grasp_force > 5)
        self._grasp_force_filter = MovingAverage(count=120, shape=(1, ))

        for i in GRIPPER_FINGER_INDICES:
            self._pb_client.changeDynamics(
                bodyUniqueId=self._id,
                linkIndex=i,
                lateralFriction=0.75,
                spinningFriction=0.75,
                rollingFriction=0.75,
            )

    def update_state(self):
        force = self._grasp_force_state
        self._grasp_force_filter.update(force)

    @property
    def revolute_joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._id, REVOLUTE_JOINT_INDICES)])

    @property
    def finger_joint_state(self):
        return np.array([i[0] for i in self._pb_client.getJointStates(self._id, GRIPPER_FINGER_INDICES)])

    @property
    def gripper_pose_euler(self):
        gripper_state = self.gripper_pose
        px, py, pz = gripper_state[0]
        ax, ay, az = pb.getEulerFromQuaternion(gripper_state[1])

        return np.array([px, py, pz]), np.array([ax, ay, az])

    @property
    def gripper_pose(self):
        gripper_state = self._pb_client.getLinkState(self._id, GRIPPER_INDEX)

        position, _ = self._pb_client.multiplyTransforms(
            gripper_state[0],
            gripper_state[1],
            (GRIPPER_ORIGIN_OFFSET, 0, 0),
            (0, 0, 0, 1)
        )

        return np.array(position), np.array(gripper_state[1])

    @lru_cache()
    def _joint_range(self):
        num_joints = self._pb_client.getNumJoints(self.id)
        lower_limits = [self._pb_client.getJointInfo(self.id, i)[8] for i in range(num_joints)]
        upper_limits = [self._pb_client.getJointInfo(self.id, i)[9] for i in range(num_joints)]

        return np.array(lower_limits), np.array(upper_limits)

    @property
    @lru_cache()
    def revolute_joint_range(self):
        ll, ul = self._joint_range()
        return ll[REVOLUTE_JOINT_INDICES], ul[REVOLUTE_JOINT_INDICES]

    @property
    @lru_cache()
    def finger_joint_range(self):
        ll, ul = self._joint_range()
        return ll[GRIPPER_FINGER_INDICES], ul[GRIPPER_FINGER_INDICES]

    def set_gripper_pose(self, position, orientation):
        position, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (-GRIPPER_ORIGIN_OFFSET, 0, 0),
            (0, 0, 0, 1)
        )

        common_params = dict(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=GRIPPER_INDEX,
            targetPosition=position,
            targetOrientation=orientation,
            maxNumIterations=1000,
            residualThreshold=0.00001,
            jointDamping=(1e-50,) * len(MOVABLE_JOINT_INDICES),
        )

        joint_states = pb.calculateInverseKinematics(
            **common_params,
            solver=self._pb_client.IK_DLS
        )[:-2]

        ll, ul = self.revolute_joint_range
        if np.any(np.isnan(joint_states) | (ll > joint_states) | (ul < joint_states)):
            joint_states = pb.calculateInverseKinematics(
                **common_params,
                lowerLimits=list(self._joint_range()[0][MOVABLE_JOINT_INDICES]),
                upperLimits=list(self._joint_range()[1][MOVABLE_JOINT_INDICES]),
                jointRanges=(2 * math.pi, ) * len(MOVABLE_JOINT_INDICES),
                restPoses=(0, ) * len(MOVABLE_JOINT_INDICES),
                solver=self._pb_client.IK_SDLS,
            )[:-2]

        return self.set_revolute_joint_state(joint_states)

    def move_gripper_pose(self, dposition, dorientation):
        current_position, current_orientation = self.gripper_pose

        position, orientation = self._pb_client.multiplyTransforms(
            current_position,
            current_orientation,
            dposition,
            dorientation
        )

        return self.set_gripper_pose(position, orientation)

    def set_revolute_joint_state(self, joint_states):
        assert len(REVOLUTE_JOINT_INDICES) == len(joint_states)

        ll, ul = self.revolute_joint_range
        limit_joint_states = np.maximum(np.minimum(joint_states, ul), ll)

        if np.any(joint_states != limit_joint_states):
            logging.debug('Out of bound joint states')
            return self._make_revolute_joint_interrupt(self.revolute_joint_state)

        self._pb_client.setJointMotorControlArray(
            self._id,
            REVOLUTE_JOINT_INDICES,
            pb.POSITION_CONTROL,
            limit_joint_states,
            positionGains=(0.3,) * len(joint_states),
            velocityGains=(1,) * len(joint_states)
        )

        return self._make_revolute_joint_interrupt(limit_joint_states)

    def set_gripper_finger(self, open):
        if open:
            return self.open_gripper()
        else:
            return self.close_gripper()

    def open_gripper(self):
        return self.set_finger_joint_state(FINGER_JOINT_RANGE[:, 1].ravel())

    def close_gripper(self):
        return interrupts.any(self.set_finger_joint_state(FINGER_JOINT_RANGE[:, 0].ravel()), self._grasp_interrupt)

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
        interrupt = NumericStateInterrupt(target_state, lambda: self.revolute_joint_state, tolerance=1e-4)
        return interrupt

    def _make_finger_joint_interrupt(self, target_state):
        assert len(target_state) == len(GRIPPER_FINGER_INDICES)
        interrupt = NumericStateInterrupt(target_state, lambda: self.finger_joint_state, tolerance=1e-3)
        return interrupt

    @property
    def _grasp_force_state(self):
        fx, fy, fz, _, _, _ = self._pb_client.getJointState(self.id, GRIPPER_FINGER_INDICES[1])[2]
        diff = np.abs(np.sqrt(np.sum(np.array([fx, fy, fz])) ** 2) - self._gravity)

        return diff

    @property
    def grasp_force(self):
        return self._grasp_force_filter.get()

    def reset_joint_states(self):
        for i in MOVABLE_JOINT_INDICES:
            self._pb_client.resetJointState(self.id, i, 0)

    def capture_gripper_camera(self):
        return self._gripper_cam.state

    def _gripper_cam_view_calculator(self):
        position, orientation = self.gripper_pose

        offset = np.array([-0.03, 0, 0.02])

        eye, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            offset,
            (0, 0, 0, 1)
        )

        to, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            offset + (100, 0, 0),
            (0, 0, 0, 1)
        )

        up, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            offset + (0, 0, 100),
            (0, 0, 0, 1)
        )

        return eye, to, up
