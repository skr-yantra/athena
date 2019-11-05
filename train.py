import os

from comet import new_experiment, wrap_experiment

import numpy as np
import ray
import click

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

import envs


def train(environment='table-clearing-v0', iterations='1000', num_gpus='1',
          num_workers='1', render='0', comet='0', save_frequency='10'):
    ray.init()

    iterations = int(iterations)
    save_frequency = int(save_frequency)
    num_gpus = int(num_gpus)
    num_workers = int(num_workers)
    render = render == '1'
    comet = comet == '1'

    comet = new_experiment() if comet else None

    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = num_gpus
    config["num_workers"] = num_workers
    config["env_config"] = {"render": render}
    config["callbacks"] = {
        "on_episode_step": _make_episode_step_handler(None if comet is None else wrap_experiment(comet)),
        "on_episode_end": _handle_episode_end,
    }

    config["lambda"] = 0.95
    config["kl_coeff"] = 0.5
    config["vf_clip_param"] = 100.0
    config["entropy_coeff"] = 0.01

    config["train_batch_size"] = 5000
    config["sample_batch_size"] = 200
    config["sgd_minibatch_size"] = 500
    config["num_sgd_iter"] = 30

    trainer = PPOTrainer(config=config, env=environment)

    for i in range(iterations):
        result = trainer.train()
        print(pretty_print(result))

        check_point = None
        if i % save_frequency == 0 or i == iterations-1:
            check_point = trainer.save()
            print('Checkpoint saved at {}'.format(check_point))

        if comet is None:
            continue

        comet.log_current_epoch(i)

        metrics = [
            'episode_reward_max', 'episode_reward_mean', 'episode_reward_min',
            'episode_len_mean', 'episodes_total', 'timesteps_total'
        ]

        for metric in metrics:
            comet.log_metric(metric, result[metric])

        if check_point is not None:
            comet.log_asset_folder(os.path.dirname(check_point), step=i)


def _handle_episode_end(info):
    episode = info['episode']
    for k, v in episode.last_info_for().items():
        episode.custom_metrics[k] = v


def _make_episode_step_handler(c):
    def handler(info):
        episode = info["episode"]
        step = episode.length

        if step % 1000 != 0 or c is None:
            return

        obs = episode.last_raw_obs_for()
        rgb = np.array(obs)[:, :, :3]
        c.comet.log_image(rgb, name=str(episode.episode_id), overwrite=True)

    return handler


@click.command('train')
@click.argument('name', type=click.STRING)
@click.option('--config', '-c', type=(str, str), multiple=True)
def main(name, config):
    train(name, **dict(config))
