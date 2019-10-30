import math

import pybullet as pb
import numpy as np
import ray

from gym import Env
from gym.spaces import Box
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

from simulator.env.table_clearing import Environment as TableClearingEnv, Action
from training.table_clearing import RewardCalculator


class GymEnvironment(Env):

    def __init__(self, config):
        super(GymEnvironment, self).__init__()

        pb.connect(pb.GUI if 'render' in config and config['render'] else pb.DIRECT)

        self.action_space = Box(shape=(5, ), high=1., low=-1., dtype=np.float32)
        self.observation_space = Box(shape=(128, 128, 3), low=0, high=255, dtype=np.uint8)

        self._env = TableClearingEnv(realtime=False, debug=False)
        self._setup_new_episode()

    def _setup_new_episode(self):
        self._env.reset()

        gripper_position = np.random.uniform((-0.25, -0.2, 0.55), (0.25, -0.6, 1.))

        self._env.robot.set_gripper_finger(True)
        self._env.robot.set_gripper_pose(
            gripper_position, pb.getQuaternionFromEuler((math.pi / 2, math.pi / 2, 0))).spin(self._env)

        self._episode = self._env.new_episode()
        self._reward_calc = RewardCalculator(self._episode.state())

    def step(self, action):
        position_scaled = action[:3] * 0.01
        orientation_scaled = action[3] * math.pi / 180.0

        self._episode.act(Action(dpos=position_scaled, dori=(orientation_scaled, 0, 0), open_gripper=action[4] > 0))

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


def main():
    ray.init()

    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["eager"] = True
    config["eager_tracing"] = True
    config["env_config"] = {"render": True}

    trainer = PPOTrainer(config=config, env=GymEnvironment)

    while True:
        print(pretty_print(trainer.train()))


if __name__ == '__main__':
    main()
