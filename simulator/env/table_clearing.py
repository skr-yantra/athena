import math
import uuid

import pybullet as pb
import numpy as np

from . import base
from ..entity.ground import Ground
from ..entity.irb120 import IRB120, GRIPPER_FINGER_INDICES, GRIPPER_INDEX, GRIPPER_ORIGIN_OFFSET
from ..entity.table import Table
from ..entity.tray import Tray
from .. import interrupts
from ..interrupts import CollisionInterrupt, TimeoutInterrupt


class Environment(base.Environment):

    def __init__(self, *args, debug=False, **kwargs):
        self._debug = debug
        super(Environment, self).__init__(*args, **kwargs)

    def _setup(self):
        super(Environment, self)._setup()

        self._pb_client.setGravity(0, 0, -9.81)

        self._ground = Ground(self._pb_client)
        self._robot = IRB120(self._pb_client, debug=self._debug)
        self._table = Table(self._pb_client, position=(0, -0.4, 0), scale=0.5)

        self._src_tray = Tray(self._pb_client, position=(0.2, -0.4, self._table.z_end),
                              scale=0.5, color=(1, 1, 0, 1), debug=self._debug)
        self._dest_tray = Tray(self._pb_client, position=(-0.2, -0.4, self._table.z_end),
                               scale=0.5, color=(0, 1, 0, 1), debug=self._debug)

        if self._debug:
            self._pb_client.addUserDebugLine(
                (GRIPPER_ORIGIN_OFFSET, 0, 0),
                (GRIPPER_ORIGIN_OFFSET + 0.1, 0, 0),
                (1, 0, 0),
                parentObjectUniqueId=self.robot.id,
                parentLinkIndex=GRIPPER_INDEX
            )

            self._pb_client.addUserDebugLine(
                (GRIPPER_ORIGIN_OFFSET, 0, 0),
                (GRIPPER_ORIGIN_OFFSET, 0.1, 0),
                (0, 1, 0),
                parentObjectUniqueId=self.robot.id,
                parentLinkIndex=GRIPPER_INDEX
            )

            self._pb_client.addUserDebugLine(
                (GRIPPER_ORIGIN_OFFSET, 0, 0),
                (GRIPPER_ORIGIN_OFFSET, 0, 0.1),
                (0, 0, 1),
                parentObjectUniqueId=self.robot.id,
                parentLinkIndex=GRIPPER_INDEX
            )

    def step(self):
        super(Environment, self).step()
        self.robot.update_state()

        if self._debug:

            self._pb_client.addUserDebugLine(
                self.robot.gripper_pose[0],
                self.robot.gripper_pose[0] + 0.002,
                lineColorRGB=(0, 0, 0),
                lineWidth=3,
                lifeTime=10.
            )

    def new_episode(self, *args, **kwargs):
        return Episode(self, *args, **kwargs)

    @property
    def robot(self):
        return self._robot

    @property
    def src_tray(self):
        return self._src_tray

    @property
    def dest_tray(self):
        return self._dest_tray


class Action(object):

    def __init__(self, dx=0., dy=0., dz=0., dyaw=0., dpitch=0., droll=0., open_gripper=True,
                 x=None, y=None, z=None, yaw=None, pitch=None, roll=None):
        self._dx = dx
        self._dy = dy
        self._dz = dz

        self._dyaw = dyaw
        self._dpitch = dpitch
        self._droll = droll

        self._open_gripper = open_gripper

        self._x = x
        self._y = y
        self._z = z

        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll

    def apply(self, env: Environment):
        current_position, current_orientation = env.robot.gripper_pose

        position, orientation = env.pb_client.multiplyTransforms(
            current_position,
            current_orientation,
            (self._dx, self._dy, self._dz),
            env.pb_client.getQuaternionFromEuler((self._dyaw, self._dpitch, self._droll))
        )

        if self._x is not None:
            position[0] = self._x

        if self._y is not None:
            position[1] = self._y

        if self._z is not None:
            position[3] = self._z

        orientation = np.array(env.pb_client.getEulerFromQuaternion(orientation))

        if self._yaw is not None:
            orientation[0] = self._yaw

        if self._pitch is not None:
            orientation[1] = self._pitch

        if self._roll is not None:
            orientation[2] = self._roll

        move_interrupt = env.robot.set_gripper_pose(position, env.pb_client.getQuaternionFromEuler(orientation))
        finger_interrupt = env.robot.set_gripper_finger(self._open_gripper)

        return interrupts.all(move_interrupt, finger_interrupt)


class Episode(object):

    _env: Environment

    def __init__(self, env, target_position=None, target_orientation=None):
        self._id = uuid.uuid4()
        self._env = env
        self._start_time = self._env.time
        self._num_actions = 0

        if target_position is None or target_orientation is None:
            target_position, target_orientation = self._generate_target_pose()

        self._target = self._env.src_tray.add_cube(target_position, target_orientation)
        self._collision_interrupt = CollisionInterrupt(self._env.robot.id, [self._target.id])

    def _generate_target_pose(self):
        tray = self._env.src_tray

        x_range = tray.x_span / 2 - 0.025
        y_range = tray.y_span / 2 - 0.025
        z_min = 0.05
        z_max = 0.15

        position = np.random.uniform([-x_range, -y_range, z_min], [x_range, y_range, z_max])
        orientation = np.random.uniform([0, 0, 0], [2*math.pi] * 3)

        return position, pb.getQuaternionFromEuler(orientation)

    def act(self, action: Action, timeout=5.):
        self._num_actions += 1
        interrupt = action.apply(self._env)
        interrupt = interrupts.any(self._collision_interrupt, TimeoutInterrupt(self.env, timeout), interrupt)
        interrupt.spin(self._env)

    def state(self):
        return EpisodeState(self)

    @property
    def id(self):
        return self._id

    @property
    def env(self):
        return self._env

    @property
    def target(self):
        return self._target

    def cleanup(self):
        return self._target.remove()

    @property
    def start_time(self):
        return self._start_time

    @property
    def num_actions(self):
        return self._num_actions


class EpisodeState(object):

    def __init__(self, episode: Episode):
        self._episode = episode

        self._gripper_pos, _ = self._episode.env.robot.gripper_pose
        self._target_pos = self._episode.target.position

        self._d_target_gripper = self._calc_d_target_gripper()
        self._d_gripper_src_tray = self._calc_d_gripper_src_tray()
        self._d_gripper_dest_tray = self._calc_d_gripper_dest_tray()

        self._grasped = self._calc_grasped()
        self._collided = self._calc_collided()

        self._gripper_cam = self._episode.env.robot.capture_gripper_camera()
        self._time = episode.env.time

        self._reached_src_tray = self._calc_reached_src_tray()
        self._reached_dest_tray = self._calc_reached_dest_tray()
        self._done = self._calc_done()

    def _calc_d_target_gripper(self):
        return np.linalg.norm(self._gripper_pos - self._target_pos)

    def _calc_d_gripper_src_tray(self):
        return np.linalg.norm(np.array(self._gripper_pos) - self._episode.env.src_tray.position)

    def _calc_d_gripper_dest_tray(self):
        return np.linalg.norm(self._gripper_pos - self._episode.env.dest_tray.position)

    def _calc_grasped(self):
        contact_f1 = len(self._episode.env.pb_client.getContactPoints(
            bodyA=self._episode.env.robot.id,
            bodyB=self._episode.target.id,
            linkIndexA=GRIPPER_FINGER_INDICES[0]
        )) > 0

        contact_f2 = len(self._episode.env.pb_client.getContactPoints(
            bodyA=self._episode.env.robot.id,
            bodyB=self._episode.target.id,
            linkIndexA=GRIPPER_FINGER_INDICES[1]
        )) > 0

        target_min_z = min(self._episode.target.z_start, self._episode.target.z_end)
        raised = target_min_z - self._episode.env.src_tray.z_start > 0.01

        other_contacts = len([i for i in self._episode.env.pb_client.getContactPoints(
            bodyA=self._episode.target.id) if i[2] != self._episode.env.robot.id]) > 0

        return contact_f1 and contact_f2 and raised and not other_contacts

    def _calc_collided(self):
        points = self._episode.env.robot.contact_points()
        exceptions = [self._episode.target.id]
        collisions = [p[2] for p in points if p[2] not in exceptions]
        return len(collisions) > 0

    def _calc_reached_src_tray(self):
        src_tray = self._episode.env.src_tray
        return self._d_gripper_src_tray < (min(src_tray.x_span, src_tray.y_span) / 2.0) - 0.01

    def _calc_reached_dest_tray(self):
        dest_tray = self._episode.env.dest_tray
        return self._d_gripper_dest_tray < (min(dest_tray.x_span, dest_tray.y_span) / 2.0) - 0.01

    def _calc_done(self):
        return self._collided or self._reached_dest_tray

    @property
    def gripper_pos(self):
        return self._gripper_pos

    @property
    def d_target_gripper(self):
        return self._d_target_gripper

    @property
    def d_gripper_src_tray(self):
        return self._d_gripper_src_tray

    @property
    def d_gripper_dest_tray(self):
        return self._d_gripper_dest_tray

    @property
    def grasped(self):
        return self._grasped

    @property
    def collided(self):
        return self._collided

    @property
    def reached_src_tray(self):
        return self._reached_src_tray

    @property
    def reached_dest_tray(self):
        return self._reached_dest_tray

    @property
    def done(self):
        return self._done

    @property
    def gripper_camera(self):
        return self._gripper_cam

    @property
    def time(self):
        return self._time

    def __str__(self):
        return '%s: %s' % (self.__class__, {
            'd_tg': self.d_target_gripper, 'd_gd': self.d_gripper_dest_tray, 'grasped': self.grasped, 'collided': self.collided
        })
