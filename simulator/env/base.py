import time

import cv2
import pybullet as pb

from ..sensors.camera import Camera


class Environment(object):

    def __init__(self, pb_client=pb, step_size=1./240., realtime=True):
        self._pb_client = pb_client

        self._step_size = step_size
        self._last_step_time = time.time()
        self._realtime = realtime

        self._setup()
        self._timer = 0.0

        self._camera = Camera(
            pb_client,
            view_calculator=lambda: ((2, 2, 2), (-2, -2, -2), (0, 0, 10)),
            resolution=(600, 600)
        )

    def spin(self, count=None):
        while count is None or count > 0:
            self.step()
            count = count - 1 if count is not None else None

    def step(self):
        self._pb_client.stepSimulation()
        self._timer += self._step_size
        current = time.time()
        elapsed = current - self._last_step_time
        to_sleep = max(0., self._step_size - elapsed)

        if self._realtime and to_sleep > 0:
            time.sleep(to_sleep)

        self._last_step_time = current + to_sleep

    def _setup(self):
        self._step = 0

    def reset(self):
        self._pb_client.resetSimulation()
        self._setup()

    def camera(self):
        return self._camera

    @property
    def pb_client(self):
        return self._pb_client

    @property
    def time(self):
        return self._timer
