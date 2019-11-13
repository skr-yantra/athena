#!/usr/bin/env python3
import click

from env_test import main as env_test
from env_benchmark import main as env_benchmark
from train import main as train
from evaluate import main as evaluate


@click.group()
def main():
    pass


main.add_command(env_test)
main.add_command(env_benchmark)
main.add_command(train)
main.add_command(evaluate)

if __name__ == '__main__':
    main()
