import os

from comet import new_experiment, new_rpc_experiment_logger

import numpy as np
import ray
import click

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.apex import ApexDDPGTrainer, APEX_DDPG_DEFAULT_CONFIG
from ray.rllib.agents.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.rllib.models import MODEL_DEFAULTS

import envs
import models


def train(environment='table-clearing-v0', iterations='1000', num_gpus='1', checkpoint=None,
          num_workers='1', render='0', comet='0', save_frequency='10', algorithm='PPO', config_trainer={}):
    ray.init()

    iterations = int(iterations)
    save_frequency = int(save_frequency)
    num_gpus = int(num_gpus)
    num_workers = int(num_workers)
    render = render == '1'
    comet = comet == '1'

    comet = new_experiment(disabled=comet is None)
    comet_rpc_server, comet_client_gen = new_rpc_experiment_logger(comet, 'localhost', 8089)

    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "env_config": {
            "render": render
        },
        "callbacks": {
            "on_episode_step": _make_episode_step_handler(comet_client_gen()),
            "on_episode_end": _handle_episode_end,
        }
    }

    trainer = _get_trainer(algorithm, environment, config, config_trainer)

    if checkpoint is not None:
        trainer.restore(checkpoint)

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


def _get_trainer(name, env, defconfig, config_trainer):
    if name == 'PPO':
        return _trainer_ppo(env, defconfig, **config_trainer)
    elif name == 'DDPG':
        return _trainer_ddpg(env, defconfig, **config_trainer)
    elif name == 'APEX_DDPG':
        return _trainer_apex_ddpg(env, defconfig, **config_trainer)
    else:
        raise Exception('unknown algorithm {}'.format(name))


def _trainer_ddpg(env, defconfig, model='svggnet_v1'):
    config = DDPG_DEFAULT_CONFIG.copy()
    _copy_dict(defconfig, config)

    config["use_state_preprocessor"] = True

    config["model"] = {
        "custom_model": model,
        "custom_options": {}
    }

    trainer = DDPGTrainer(config=config, env=env)

    return trainer


def _trainer_apex_ddpg(env, defconfig, model='svggnet_v1'):
    config = APEX_DDPG_DEFAULT_CONFIG.copy()
    _copy_dict(defconfig, config)

    config["use_state_preprocessor"] = True

    config["model"] = {
        "custom_model": model,
        "custom_options": {}
    }

    trainer = ApexDDPGTrainer(config=config, env=env)

    return trainer


def _trainer_ppo(env, defconfig, model='svggnet_v1'):
    config = PPO_DEFAULT_CONFIG.copy()
    _copy_dict(defconfig, config)

    config["lambda"] = 0.95
    config["kl_coeff"] = 0.5
    config["vf_clip_param"] = 100.0
    config["entropy_coeff"] = 0.01

    config["train_batch_size"] = 5000
    config["sample_batch_size"] = 200
    config["sgd_minibatch_size"] = 500
    config["num_sgd_iter"] = 30

    config["model"] = {
        "custom_model": model,
        "custom_options": {}
    }

    trainer = PPOTrainer(config=config, env=env)

    return trainer


def _handle_episode_end(info):
    episode = info['episode']
    _flatten_info(episode.last_info_for(), episode.custom_metrics)


def _flatten_info(info, out, prefix=None):
    for k, v in info.items():
        name = '{}_{}'.format(prefix, k) if prefix is not None else k
        if isinstance(v, dict):
            _flatten_info(v, out, name)
        else:
            out[name] = v


def _make_episode_step_handler(c):
    def handler(info):
        episode = info["episode"]
        step = episode.length

        if step % 1000 != 0 or c is None:
            return

        obs = episode.last_raw_obs_for()
        rgb = np.array(obs)[:, :, :3]
        c.log_image(rgb.tolist(), name=str(episode.episode_id), overwrite=True)

    return handler


def _copy_dict(src, dest):
    for k, v in src.items():
        dest[k] = v


@click.command('train')
@click.argument('name', type=click.STRING)
@click.option('--config', '-c', type=(str, str), multiple=True)
@click.option('--config_trainer', '-ct', type=(str, str), multiple=True)
def main(name, config, config_trainer):
    train(name, **dict(config), config_trainer=dict(config_trainer))
