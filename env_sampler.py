import math

import click
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder

import gym
import envs


def sample_episode(episode_id, env, batch_builder, max_actions=10000):

    s_tm1 = env.reset()
    a_tm1 = np.zeros_like(env.action_space.sample())
    r_tm1 = 0

    for i in range(max_actions):
        a_t = env.action_space.sample()
        s_t, r_t, done, info = env.step(a_t)

        batch_builder.add_values(
            t=i,
            eps_id=episode_id,
            agent_index=0,
            obs=s_tm1,
            actions=a_t,
            rewards=r_t,
            prev_actions=a_tm1,
            prev_rewards=r_tm1,
            dones=done,
            infos=info,
            new_obs=s_t,
            weights=1.0,
        )

        s_tm1 = s_t
        a_tm1 = a_t
        r_tm1 = r_t

        if done:
            break


def sample_env_worker(worker_index, env_name, episodes, output_path):
    env = gym.make(env_name, config={})
    batch_builder = SampleBatchBuilder()
    json_writer = JsonWriter(output_path, ioctx=IOContext(worker_index=worker_index))
    for episode in tqdm(episodes, position=worker_index):
        sample_episode(episode, env, batch_builder)
        json_writer.write(batch_builder.build_and_reset())


def generate(env_name, output_dir, num_episodes, num_workers=8):
    episodes_per_worker = math.ceil(num_episodes / num_workers)
    episodes_split = [np.arange(episodes_per_worker) + (i * episodes_per_worker) for i in range(num_workers)]

    Parallel(n_jobs=num_workers)(delayed(sample_env_worker)(
        i, env_name, episodes, output_dir) for i, episodes in enumerate(episodes_split))


@click.command('env_sampler')
@click.argument('name', type=str)
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False, writable=True))
@click.option('--num_episodes', '-n', type=int, default=100)
def main(name, output_dir, num_episodes):
    generate(name, output_dir, num_episodes)
