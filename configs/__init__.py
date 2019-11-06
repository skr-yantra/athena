import os

from attrdict import AttrDict
import yaml


def read_params(name):
    path = os.path.join(os.path.dirname(os.path.relpath(__file__)), '%s.yaml' % (name, ))

    with open(path) as f:
        data = yaml.safe_load(f)
        return AttrDict(data)
