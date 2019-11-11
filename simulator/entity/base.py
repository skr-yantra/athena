import pybullet as pb
import numpy as np


class Entity(object):

    def __init__(self, urdf, pb_client=pb, position=(0, 0, 0), orientation=(0, 0, 0, 1),
                 fixed_base=False, scale=1, debug=False):
        self._debug = debug
        self._pb_client = pb_client
        self._id = pb_client.loadURDF(
            urdf,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=fixed_base,
            globalScaling=scale
        )
        self._position = position
        self._orientation = orientation

        if debug:
            self._pb_client.addUserDebugLine(
                (0, 0, 0),
                (0, 0, 0.1),
                (1, 0, 0),
                parentObjectUniqueId=self.id
            )

            self._pb_client.addUserDebugLine(
                (0, 0, 0),
                (0, 0.1, 0),
                (0, 1, 0),
                parentObjectUniqueId=self.id
            )

            self._pb_client.addUserDebugLine(
                (0, 0, 0),
                (0.1, 0, 0),
                (0, 0, 1),
                parentObjectUniqueId=self.id
            )

    @property
    def id(self):
        return self._id

    @property
    def pose(self):
        pos, orientation = self._pb_client.getBasePositionAndOrientation(self._id)
        return np.array(pos), np.array(orientation)

    @property
    def position(self):
        return self.pose[0]

    @property
    def orientation(self):
        return self.pose[1]

    @property
    def orientation_euler(self):
        return self._pb_client.getEulerFromQuaternion(self.orientation)

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

    def transform(self, position, orientation):
        return self._pb_client.multiplyTransforms(self.position, self.orientation, position, orientation)

    def contact_points(self):
        return self._pb_client.getContactPoints(self._id)

    def remove(self):
        self._pb_client.removeBody(self._id)
