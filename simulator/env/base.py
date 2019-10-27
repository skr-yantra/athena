import time

import pybullet as pb


class Environment(object):

    def __init__(self, pb_client=pb, step_size=1./240., realtime=True):
        self._pb_client = pb_client

        self._step_size = step_size * 10e9
        self._last_step_time = time.time_ns()
        self._realtime = realtime

        self._setup()
        self._timer = 0.0

    def spin(self):
        while True:
            self.step()

    def step(self):
        self._pb_client.stepSimulation()
        self._timer += self._step_size
        current = time.time_ns()
        elapsed = current - self._last_step_time
        to_sleep = max(0., self._step_size - elapsed)

        if self._realtime and to_sleep > 0:
            time.sleep(to_sleep / 10e9)

        self._last_step_time = current + to_sleep

    def _setup(self):
        self._step = 0

    def reset(self):
        self._pb_client.resetSimulation()
        self._setup()

    @property
    def pb_client(self):
        return self._pb_client

    @property
    def time(self):
        return self._timer
