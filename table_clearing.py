import math

import click
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

    def __init__(self, config, target_pose=None, gripper_pose=None):
        super(GymEnvironment, self).__init__()

        pb.connect(pb.GUI if 'render' in config and config['render'] else pb.DIRECT)

        self._target_pose = target_pose
        self._gripper_pose = gripper_pose

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
        action[:3] = action[:3] * 0.01
        action[3] = action[3] * math.pi / 180.0

        self._episode.act(Action(
            dx=action[0], dy=action[1], dz=action[2],
            droll=action[3], pitch=math.pi/2, yaw=math.pi/2,
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


def train(num_gpus=0, num_workers=1, render=False):
    ray.init()

    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = num_gpus
    config["num_workers"] = num_workers
    config["env_config"] = {"render": render}

    trainer = PPOTrainer(config=config, env=GymEnvironment)

    while True:
        print(pretty_print(trainer.train()))


def test_environment():
    env = GymEnvironment(
        config={'render': True, 'realtime': True, 'debug': True},
        target_pose=((0, 0, 0.1), (0, 0, 0, 1)),
        gripper_pose=((0.2, -0.5, 0.7), pb.getQuaternionFromEuler((math.pi / 2., math.pi / 2., math.pi / 3.)))
    )
    env._env.spin()


@click.group()
def cli():
    pass


@cli.command('train')
def _cli_train():
    train()


@cli.command('test_environment')
def _cli_test_environment():
    test_environment()


if __name__ == '__main__':
    cli()
