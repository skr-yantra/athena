import math
import logging

import click
import pybullet as pb
import pybullet_data
import numpy as np
import gym
import envs

from simulator.env.irb120 import IRB120


def test_table_clearing_v0(realtime='0', debug='0'):
    config = {
        'render': True,
        'realtime': realtime == '1',
        'debug': debug == '1',
        'target_pose': ((0, 0, 0.1), pb.getQuaternionFromEuler((0, 0, math.pi / 4))),
        'gripper_pose': ((0.2, -0.4, 0.7), pb.getQuaternionFromEuler((math.pi / 2., math.pi / 2., 0)))
    }

    env = gym.make('table-clearing-v0', config=config)

    def act(action, count=1):
        for i in range(count):
            _, reward, done, info = env.step(np.array(action)*100.)
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
    act([1., 0., 0., 0., 0.], 13)
    act([0., 0., 0., 0., 1.], 1)
    act([-1., 0., 0., 0., 1.], 2)

    env._env.spin()


def test_pick_and_place_manual():
    pb.connect(pb.GUI)
    simulator = IRB120()

    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("table/table.urdf", [0.6, 0, 0], globalScaling=0.5)
    pb.loadURDF("cube_small.urdf", [0.4, 0, 0.5], globalScaling=0.5)

    simulator.release_gripper()
    simulator.move_absolute([0.445, 0, 0.5, 0, math.pi / 2, 0])
    simulator.move_relative([0, 0, -0.14, 0, 0, 0])
    simulator.hold_gripper()
    simulator.move_relative([0, 0, 0.1, 0, 0, 0])
    simulator.move_relative([0.2, 0, 0, 0, 0, 0])
    simulator.move_relative([0, 0, -0.1, 0, 0, 0])
    simulator.release_gripper()
    simulator.move_relative([0, 0, 0.1, 0, 0, 0])
    simulator.move_relative([-0.2, 0, 0, 0, 0, 0])
    simulator.spin()


_ENVIRONMENT_REGISTRY = {
    'table-clearing-v0': test_table_clearing_v0,
    'pick-and-place-manual': test_pick_and_place_manual
}


@click.command('env_test')
@click.argument('name', type=click.Choice(_ENVIRONMENT_REGISTRY.keys(), case_sensitive=False))
@click.option('--config', '-c', type=(str, str), multiple=True)
def main(name, config):
    _ENVIRONMENT_REGISTRY[name](**dict(config))
