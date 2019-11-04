import os

from comet import new_experiment

import ray
import click

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

import envs


def train(environment='table-clearing-v0', iterations='1000', num_gpus='1',
          num_workers='1', render='0', comet='0', save_every='10'):
    ray.init()

    iterations = int(iterations)
    save_every = int(save_every)
    num_gpus = int(num_gpus)
    num_workers = int(num_workers)
    render = render == '1'
    comet = comet == '1'

    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = num_gpus
    config["num_workers"] = num_workers
    config["env_config"] = {"render": render}

    comet = new_experiment() if comet else None

    trainer = PPOTrainer(config=config, env=environment)

    for i in range(iterations):
        result = trainer.train()
        print(pretty_print(result))

        if i % save_every == 0 or i == iterations-1:
            check_point = trainer.save()
            print('Checkpoint saved at {}'.format(check_point))

            if comet is not None:
                comet.log_asset_folder(os.path.dirname(check_point), step=i)

        if comet is None:
            continue

        comet.log_current_epoch(i)

        metrics = [
            'episode_reward_max', 'episode_reward_mean', 'episode_reward_min',
            'episode_len_mean', 'episodes_total', 'timesteps_total'
        ]

        for metric in metrics:
            comet.log_metric(metric, result[metric])


@click.command('train')
@click.argument('name', type=click.STRING)
@click.option('--config', '-c', type=(str, str), multiple=True)
def main(name, config):
    train(name, **dict(config))
