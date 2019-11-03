import os

import numpy as np


def assert_exist(file):
    assert os.path.exists(file)


def unimplemented():
    raise Exception('not implemented')


def clip_line_end(start, end, length=0.01):
    vec = np.array(end) - np.array(start)
    dir = vec / np.sqrt(np.sum(vec**2))
    clipped = dir * length

    return clipped + start
