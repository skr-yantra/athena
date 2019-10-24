import math

import pybullet as pb
import numpy as np

from . import base
from ..entity.ground import Ground
from ..entity.irb120 import IRB120, GRIPPER_FINGER_INDICES
from ..entity.table import Table
from ..entity.tray import Tray
from .. import interrupts
from ..interrupts import CollisionInterrupt


class Environment(base.Environment):

    def _setup(self):
        super(Environment, self)._setup()

        self._pb_client.setGravity(0, 0, -9.81)

        self._ground = Ground(self._pb_client)
        self._robot = IRB120(self._pb_client)
        self._src_table = Table(self._pb_client, position=(0, -0.5, 0), scale=0.5)
        self._dest_table = Table(self._pb_client, position=(0, 0.5, 0), scale=0.5)

        self._src_tray = Tray(self._pb_client, position=(0, -0.5, self._src_table.z_end), scale=0.5)
        self._dest_tray = Tray(self._pb_client, position=(0, 0.5, self._dest_table.z_end), scale=0.5)

    def new_episode(self):
        return Episode(self)

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

    def __init__(self, dpos, dori, open_gripper):
        self._dpos = dpos
        self._dori = dori
        self._open_gripper = open_gripper

    @property
    def dposition(self):
        return self._dpos

    @property
    def dorientation(self):
        return self._dori

    @property
    def open_gripper(self):
        return self._open_gripper


class Episode(object):

    _env: Environment

    def __init__(self, env, target_position=None, target_orientation=None):
        self._env = env

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

    def act(self, action: Action):
        dpos, dori, open_gripper = action.dposition, action.dorientation, action.open_gripper

        move_interrupt = self._env.robot.move_gripper_pose(dpos, pb.getQuaternionFromEuler(dori))
        gripper_interrupt = self._env.robot.set_gripper_finger(open_gripper)

        interrupt = interrupts.all(move_interrupt, gripper_interrupt)
        interrupt = interrupts.any(self._collision_interrupt, interrupt)
        interrupt.spin(self._env)

        return EpisodeState(self)

    @property
    def env(self):
        return self._env

    @property
    def target(self):
        return self._target


class EpisodeState(object):

    def __init__(self, episode: Episode):
        self._episode = episode

        self._gripper_pos, _ = self._episode.env.robot.gripper_pose
        self._target_pos = self._episode.target.position

        self._d_tg = self._calc_d_tg()
        self._d_gd = self._calc_d_gd()
        self._grasped = self._calc_grasped()
        self._collided = self._calc_collided()

    def _calc_d_tg(self):
        return np.linalg.norm([self._gripper_pos, self._target_pos])

    def _calc_d_gd(self):
        return np.linalg.norm([self._gripper_pos, self._episode.env.dest_tray.position])

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

    @property
    def d_tg(self):
        return self._d_tg

    @property
    def d_gd(self):
        return self._d_gd

    @property
    def grasped(self):
        return self._grasped

    @property
    def collided(self):
        return self._collided

    def __str__(self):
        return '%s: %s' % (self.__class__, {
            'd_tg': self.d_tg, 'd_gd': self.d_gd, 'grasped': self.grasped, 'collided': self.collided
        })
