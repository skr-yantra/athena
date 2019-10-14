import pybullet as pb
import numpy as np


class Entity(object):

    def __init__(self, urdf, pb_client=pb, pose=(0, 0, 0, 0, 0, 0), fixed_base=False, scale=1):
        self._pb_client = pb_client
        self._id = pb_client.loadURDF(
            urdf,
            basePosition=pose[:3],
            baseOrientation=pb_client.getQuaternionFromEuler(pose[3:]),
            useFixedBase=fixed_base,
            globalScaling=scale
        )
        self._pose = pose

    @property
    def pose(self):
        return self._pose

    @property
    def position(self):
        return self.pose[:3]

    @property
    def orientation(self):
        return self.pose[3:]

    @property
    def orientation_quaternion(self):
        return self._pb_client.getQuaternionFromEuler(self.orientation)

    @property
    def bounding_box(self):
        start, end = self._pb_client.getAABB(self._id)
        return np.array(start), np.array(end)

    @property
    def x_start(self):
        return self.bounding_box[0][0]

    @property
    def x_end(self):
        return self.bounding_box[0][1]

    @property
    def y_start(self):
        return self.bounding_box[0][1]

    @property
    def y_end(self):
        return self.bounding_box[1][1]

    @property
    def z_start(self):
        return self.bounding_box[0][2]

    @property
    def z_end(self):
        return self.bounding_box[1][2]

    @property
    def x_span(self):
        return self._axis_span(0)

    @property
    def y_span(self):
        return self._axis_span(1)

    @property
    def z_span(self):
        return self._axis_span(2)

    def _axis_span(self, axis):
        start, end = self.bounding_box
        return end[axis] - start[axis]