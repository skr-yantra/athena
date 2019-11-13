import time
import click

import tqdm
import numpy as np
import gym
import pandas as pd

import envs


def test_gym_env_action(name, render='1', realtime='0', debug='0', num_episodes='100', max_actions='100'):
    render = render == '1'
    realtime = realtime == '1'
    debug = debug == '1'
    num_episodes = int(num_episodes)
    max_actions = int(max_actions)

    config = {
        'render': render,
        'realtime': realtime,
        'debug': debug,
    }

    env = gym.make(name, config=config)

    def _run_episode():
        env.reset()

        exec_times = []
        for i in range(max_actions):
            start = time.process_time_ns()
            _, _, done, _ = env.step(action=env.action_space.sample())
            exec_times.append(time.process_time_ns() - start)

            if done:
                break

        exec_times = np.array(exec_times) / 1e9
        return exec_times

    episode_times = []
    it = tqdm.tqdm(range(num_episodes))
    for _ in it:
        exec_times = _run_episode()
        total = np.sum(exec_times)
        mean = np.mean(exec_times)
        max = np.max(exec_times)
        min = np.min(exec_times)

        episode_times.append([total, mean, min, max, exec_times.shape[0]])

        it.write('Executed {} actions in {}s with mean action time {}s'.format(
            exec_times.shape[0], total, mean
        ))

    episode_times = np.array(episode_times)
    print(pd.DataFrame(episode_times, columns=[
        'Total Episode Time', 'Mean Action Time', 'Max Action Time', 'Min Action Time', '# of Actions'
    ]).describe())


@click.command('env_benchmark')
@click.argument('name')
@click.option('--config', '-c', type=(str, str), multiple=True)
def main(name, config):
    test_gym_env_action(name, **dict(config))
