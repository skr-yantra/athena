import ray
import click

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.rollout import rollout

import envs


def evaluate(env, model):
    ray.init()
    config = DEFAULT_CONFIG.copy()
    config["env_config"] = {"render": True}
    config["num_workers"] = 1

    trainer = PPOTrainer(config=config, env=env)
    trainer.restore(model)

    rollout(trainer, env, 1000)


@click.command('evaluate')
@click.argument('name', type=str)
@click.argument('model', type=click.Path(exists=True))
def main(name, model):
    evaluate(name, model)
