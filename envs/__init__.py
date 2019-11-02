from ray.tune.registry import register_env as register_ray
from gym.envs.registration import register as register_gym

from .table_clearing import GymEnvironment as TableClearingEnvironment


_ENVIRONMENTS = {
    'table-clearing-v0': TableClearingEnvironment
}


def make_ray_env_gen(name):
    def creator(config):
        env = _ENVIRONMENTS[name]
        return env(config)

    return creator


for name, env in _ENVIRONMENTS.items():
    register_ray(name, make_ray_env_gen(name))
    register_gym(id=name, entry_point=make_ray_env_gen(name))
