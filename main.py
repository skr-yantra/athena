import click

from env_test import main as env_test
from train import main as train


@click.group()
def main():
    pass


main.add_command(env_test)
main.add_command(train)

if __name__ == '__main__':
    main()
