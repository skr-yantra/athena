import math

import pybullet as pb
import numpy as np
from gym import Env
from gym.spaces import Box

from hparams import read_params
from simulator.env.table_clearing import EpisodeState, Action
from simulator.env.table_clearing import Environment as TableClearingEnv, Action


class GymEnvironment(Env):

    def __init__(self, config):
        super(GymEnvironment, self).__init__()

        pb.connect(pb.GUI if 'render' in config and config['render'] else pb.DIRECT)

        self._target_pose = config['target_pose'] if 'target_pose' in config else None
        self._gripper_pose = config['gripper_pose'] if 'gripper_pose' in config else None

        self.action_space = Box(shape=(5, ), high=1., low=-1., dtype=np.float32)
        self.observation_space = Box(shape=(128, 128, 3), low=0, high=255, dtype=np.uint8)

        realtime = 'realtime' in config and config['realtime']
        debug = 'debug' in config and config['debug']
        self._env = TableClearingEnv(realtime=realtime, debug=debug)
        self._episode = None

        self._setup_new_episode()

    def _setup_new_episode(self):
        if self._episode is not None:
            self._episode.cleanup()

        gripper_position = np.random.uniform((-0.25, -0.2, 0.55), (0.25, -0.6, 1.))
        gripper_orientation = pb.getQuaternionFromEuler((math.pi/2, math.pi/2, np.random.uniform(0, 2.0 * math.pi)))

        if self._gripper_pose is not None:
            gripper_position, gripper_orientation = self._gripper_pose

        target_position, target_orientation = None, None
        if self._target_pose is not None:
            target_position, target_orientation = self._target_pose

        self._env.robot.set_gripper_finger(True)
        self._env.robot.set_gripper_pose(gripper_position, gripper_orientation).spin(self._env)

        self._episode = self._env.new_episode(target_position=target_position, target_orientation=target_orientation)
        self._reward_calc = RewardCalculator(self._episode.state())

    def step(self, action):
        action = np.array(action)
        action[:3] = action[:3] * 0.01
        action[3] = 10. * action[3] * math.pi / 180.0

        self._episode.act(Action(
            dx=action[0], dy=action[1], dz=action[2],
            dyaw=action[3], pitch=math.pi/2, yaw=None, roll=None,
            open_gripper=action[4] > 0))

        state = self._episode.state()
        reward = self._reward_calc.update(state)
        done = state.collided

        return GymEnvironment._proc_state(state), reward, done, {}

    @classmethod
    def _proc_state(cls, state):
        rgba, _ = state.gripper_camera
        return rgba[:, :, :3]

    def reset(self):
        self._setup_new_episode()
        return GymEnvironment._proc_state(self._episode.state())

    def render(self, mode='human'):
        raise Exception('Render not supported. Use render in env_config instead')


def random_action():
    dpos = np.random.uniform(-0.01, 0.01, 3)
    dori = np.random.uniform(-10, 10) * math.pi/180.0
    open = np.random.uniform(0, 1) > 0.5

    return Action(dx=dpos[0], dy=dpos[1], dz=dpos[2],
                  dyaw=dori, dpitch=0, droll=0,
                  yaw=math.pi/2, pitch=math.pi/2,
                  open_gripper=open)


class RewardCalculator(object):

    _s_tm1: EpisodeState

    def __init__(self, initial_state, params=read_params('table_clearing_base')):
        self._s_tm1 = initial_state
        self._params = params

    def update(self, state: EpisodeState):
        reward = 0.0

        # Reaching
        if not state.grasped:
            reward += self._params.reward.reaching * np.sign(self._s_tm1.d_tg - state.d_tg)

        # Delivering
        if state.grasped:
            reward += self._params.reward.delivering * np.sign(self._s_tm1.d_gd - state.d_gd)

        # Successful grasp
        if not self._s_tm1.grasped and state.grasped:
            reward += self._params.reward.grasped

        reached_destination = state.d_gd < 0.125

        # Dropped target
        if self._s_tm1.grasped and not state.grasped and not reached_destination:
            reward += self._params.reward.dropped

        # Reached destination
        if self._s_tm1.grasped and not state.grasped and reached_destination:
            reward += self._params.reward.delivered

        # Collided
        if state.collided:
            reward += self._params.reward.collided

        self._s_tm1 = state

        return reward
