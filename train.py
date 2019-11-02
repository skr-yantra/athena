import click

import envs


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


@click.command('train')
def main():
    train()
