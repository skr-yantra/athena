import os


def assert_exist(file):
    assert os.path.exists(file)


def unimplemented():
    raise Exception('not implemented')
