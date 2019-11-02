import math

import click
import pybullet as pb
import numpy as np
import gym
import training


def train(num_gpus=0, num_workers=1, render=False):
    import ray
    from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
    from ray.tune.logger import pretty_print

    ray.init()

    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = num_gpus
    config["num_workers"] = num_workers
    config["env_config"] = {"render": render}

    trainer = PPOTrainer(config=config, env='table-clearing-v0')

    while True:
        print(pretty_print(trainer.train()))


def test_environment():
    config = {
        'render': True,
        'realtime': False,
        'debug': False,
        'target_pose': ((0, 0, 0.1), pb.getQuaternionFromEuler((0, 0, math.pi / 4))),
        'gripper_pose': ((0.2, -0.4, 0.7), pb.getQuaternionFromEuler((math.pi / 2., math.pi / 2., 0)))
    }

    env = gym.make('table-clearing-v0', config=config)

    def act(action, count=1):
        for i in range(count):
            _, reward, done, info = env.step(np.array(action))
            print('action {} reward {} done {} info {}'.format(action, reward, done, info))

    act([1., 0., 0., 0., 1.], 30)
    act([1., 0., 0., 0., 1.], 3)
    act([0., 0., 0., 1., 1.], 4)
    act([0., 0., 0., 0.2, 1.], 1)
    act([1., 0., 0., 0., 1.], 4)
    act([0., 0., 0., 0., 0.], 1)
    act([-1., 0., 0., 0., 0.], 15)
    act([0., 0., 0., -1., 0.], 4)
    act([0., 0., 0., -0.2, 0.], 1)
    act([0., -1., 0., 0, 0.], 35)
    act([1., 0., 0., 0., 0.], 10)
    act([0., 0., 0., 0., 1.], 1)
    act([-1., 0., 0., 0., 1.], 10)

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
