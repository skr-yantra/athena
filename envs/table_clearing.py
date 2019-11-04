import math

import pybullet as pb
import numpy as np
from gym import Env
from gym.spaces import Box

from hparams import read_params
from simulator.env.table_clearing import EpisodeState, Action
from simulator.env.table_clearing import Environment as TableClearingEnv, Action


_TIME_LIMIT = 30 * 60 * 1e9


class GymEnvironment(Env):

    def __init__(self, config):
        super(GymEnvironment, self).__init__()
        self._parse_config(**config)

        pb.connect(pb.GUI if self._render else pb.DIRECT)

        self.action_space = Box(shape=(5, ), high=1., low=-1., dtype=np.float32)
        self.observation_space = Box(shape=(128, 128, 4), low=0, high=255, dtype=np.uint8)

        self._env = TableClearingEnv(realtime=self._realtime, debug=self._debug)
        self._episode = None

        self._setup_new_episode()

    def _parse_config(self, render=False, realtime=False, debug=False, target_pose=None, gripper_pose=None):
        self._render = render
        self._realtime = realtime
        self._debug = debug
        self._target_pose = target_pose
        self._gripper_pose = gripper_pose

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

        self._env.robot.reset_joint_states()
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

        elapsed_time = self._episode.env.time - self._episode.start_time

        done = state.done or elapsed_time > _TIME_LIMIT

        return GymEnvironment._proc_state(state), reward, done, {}

    @classmethod
    def _proc_state(cls, state):
        rgba, depth = state.gripper_camera
        obs = np.zeros((128, 128, 4))

        obs[:, :, :3] = rgba[:, :, :3]
        obs[:, :, 3] = depth

        return obs

    def reset(self):
        self._setup_new_episode()
        return GymEnvironment._proc_state(self._episode.state())

    def render(self, mode='human'):
        raise Exception('Render not supported. Use render in env_config instead')


class RewardCalculator(object):

    _s_tm1: EpisodeState

    def __init__(self, initial_state, params=read_params('table_clearing_base')):
        self._s_tm1 = initial_state
        self._params = params

    def update(self, state: EpisodeState):
        run_time = (state.time - self._s_tm1.time) / 1e9

        # Time penalty
        reward = self._params.reward.time * run_time

        # Action penalty
        reward += self._params.reward.action_penalty

        # Entered src tray (not grasped)
        if state.reached_src_tray and not self._s_tm1.reached_src_tray and not state.grasped:
            reward += self._params.reward.enter_src_tray_not_grasped

        # Exited src tray (not grasped)
        if self._s_tm1.reached_src_tray and not state.reached_src_tray and not state.grasped:
            reward += self._params.reward.exit_src_tray_not_grasped

        # Successful grasp
        if not self._s_tm1.grasped and state.grasped:
            reward += self._params.reward.grasped

        # Dropped target
        if self._s_tm1.grasped and not state.grasped and not state.reached_dest_tray:
            reward += self._params.reward.dropped

        # Reached destination
        if self._s_tm1.grasped and not state.grasped and state.reached_dest_tray:
            reward += self._params.reward.delivered

        # Collided
        if state.collided:
            reward += self._params.reward.collided

        self._s_tm1 = state

        return reward
