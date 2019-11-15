import logging
import math

import pybullet as pb
import numpy as np
from gym import Env
from gym.spaces import Box

from configs import read_params
from simulator.env.table_clearing import EpisodeState, Episode
from simulator.env.table_clearing import Environment as TableClearingEnv, Action
from .reward import RewardLog, RewardSession


class GymEnvironment(Env):

    _episode: Episode

    def __init__(self, config):
        super(GymEnvironment, self).__init__()
        self._parse_config(**config)

        pb.connect(pb.GUI if self._render else pb.DIRECT)

        self.action_space = Box(shape=(5, ), high=1., low=-1., dtype=np.float32)
        self.observation_space = Box(shape=(84, 84, 4), low=0, high=1, dtype=np.float32)

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
        self._env.robot.open_gripper()
        self._env.robot.set_gripper_pose(gripper_position, gripper_orientation).spin(self._env)

        self._episode = self._env.new_episode(target_position=target_position, target_orientation=target_orientation)
        self._reward_calc = RewardCalculator(self._episode.state())

        self._env.spin(240)

        def spin(interrupt):
            interrupt.spin(self._env, max_time=10.)

        def state_initial():
            spin(self._approach_safe_zone())

        def state_approached_grasp_zone():
            state_initial()
            spin(self._approach_grasp_zone())

        def state_approach_target():
            state_approached_grasp_zone()
            spin(self._approach_target())

        def state_held():
            state_approach_target()
            spin(self._env.robot.close_gripper())
            spin(self._approach_target())

        def state_grasped():
            state_held()
            spin(self._approach_grasp_zone())

        def state_grasped_safe_zone():
            state_grasped()
            spin(self._approach_safe_zone())

        state_choices = [
            state_initial,
            state_approached_grasp_zone,
            state_approach_target,
            state_held,
            state_grasped,
            state_grasped_safe_zone
        ]

        def init_env_state(state):
            state()

            if self._episode.state().collided:
                self._setup_new_episode()

        if self._target_pose is None:
            state = state_choices[np.random.choice(len(state_choices))]
            logging.info("Initializing environment with state {}".format(state.__name__))
            init_env_state(state)

    def _approach_grasp_zone(self):
        target_position = self._episode.target.position
        target_orientation = self._episode.target.orientation_euler

        gripper_approach_position = target_position + [0, 0, 0.15]
        gripper_approach_orientation = self._env.pb_client.getQuaternionFromEuler(
            (math.pi/2, math.pi/2, target_orientation[2])
        )

        return self._env.robot.set_gripper_pose(
            gripper_approach_position,
            gripper_approach_orientation
        )

    def _approach_target(self):
        return self._env.robot.set_gripper_pose(
            np.array([0, 0, -0.0026]) + self._episode.target.position,
            self._env.robot.gripper_pose[1]
        )

    def _approach_safe_zone(self):
        safe_min = np.array([-0.5, -0.5, 0.4])
        safe_max = np.array([0.5, -0.15, 0.8])
        return self._env.robot.set_gripper_pose(
            np.random.uniform(safe_min, safe_max),
            pb.getQuaternionFromEuler((math.pi/2, math.pi/2, np.random.uniform(0, 2.0 * math.pi)))
        )

    def step(self, action):
        if np.any(np.isnan(action)):
            print('Invalid action {}'.format(action))

        action = np.array(action)
        action[:3] = action[:3] * 0.01
        action[3] = 10. * action[3] * math.pi / 180.0

        self._episode.act(Action(
            dx=action[0], dy=action[1], dz=action[2],
            dyaw=action[3], pitch=math.pi/2, yaw=None, roll=None,
            open_gripper=action[4] > 0))

        state = self._episode.state()
        reward, done = self._reward_calc.update(self._episode, state)

        return GymEnvironment._proc_state(state), reward, done, self._reward_calc.info

    @classmethod
    def _proc_state(cls, state):
        rgba, depth = state.gripper_camera
        obs = np.zeros((rgba.shape[0], rgba.shape[1], 4))

        obs[:, :, :3] = rgba[:, :, :3] / 255.
        obs[:, :, 3] = depth / 255.

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
        self._log = RewardLog()

    def update(self, episode, state: EpisodeState):
        rewards = RewardSession()
        done = False

        run_time = state.time - self._s_tm1.time
        episode_elapsed_time = episode.env.time - episode.start_time
        episode_num_actions = episode.num_actions

        # Time penalty
        rewards.time_penalty = self._params.reward.time * run_time

        # Action penalty
        rewards.action_penalty = self._params.reward.action_penalty

        # Travel to target penalty
        if not state.grasped:
            rewards.travel_to_target_penalty = self._calc_travel_reward(
                state,
                self._s_tm1.d_target_gripper - state.d_target_gripper,
                self._params.reward.travel_to_target
            )

        # Travel to destination penalty
        if state.grasped:
            rewards.travel_to_dest_penalty = self._calc_travel_reward(
                state,
                self._s_tm1.d_gripper_dest_tray - state.d_gripper_dest_tray,
                self._params.reward.travel_to_dest
            )

        # Entered src tray (not grasped)
        if state.reached_src_tray and not self._s_tm1.reached_src_tray and not state.grasped:
            rewards.enter_src_tray_reward = self._params.reward.enter_src_tray_not_grasped

        # Exited src tray (not grasped)
        if self._s_tm1.reached_src_tray and not state.reached_src_tray and not state.grasped:
            rewards.exit_src_tray_penalty = self._params.reward.exit_src_tray_not_grasped

        # Successful grasp
        if not self._s_tm1.grasped and state.grasped:
            rewards.grasp_reward = self._params.reward.grasped

        # Dropped target
        if self._s_tm1.grasped and not state.grasped and not state.reached_dest_tray:
            rewards.drop_penalty = self._params.reward.dropped

        # Reached destination
        if self._s_tm1.grasped and not state.grasped and state.reached_dest_tray:
            rewards.reach_destination_reward = self._params.reward.delivered

        # Collided
        if state.collided:
            rewards.collision_penalty = self._params.reward.collided
            done = True

        # Timeout
        if episode_elapsed_time > self._params.episode.max_time:
            rewards.max_time_penalty = self._params.reward.max_time
            done = True

        # Max actions used
        if episode_num_actions > self._params.episode.max_actions:
            rewards.max_actions_penalty = self._params.reward.max_actions
            done = True

        self._s_tm1 = state
        self._log.log(rewards)

        total_reward = rewards.sum()
        assert np.isfinite(total_reward)

        return total_reward, done

    def _calc_travel_reward(self, state, dist_reduced, rate):
        dist_travelled = np.linalg.norm(np.array(state.gripper_pos) - self._s_tm1.gripper_pos)
        return max(dist_reduced, 0) * rate - \
            max(dist_travelled - dist_reduced, 0) * rate

    @property
    def info(self):
        return self._log.info
