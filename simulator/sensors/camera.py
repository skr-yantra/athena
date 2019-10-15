import pybullet as pb

from .base import Sensor


class Camera(Sensor):

    def __init__(self, pb_client=pb, resolution=(320, 240), fov=60, near_plane=0.01, far_plane=100.,
                 pose_reader=lambda: ((0, 0, 1), (0, 1, 0, 0)), debug=False):
        super(Camera, self).__init__(pb_client)
        self._res_x, self._res_y = resolution

        self._projection_matrix = pb_client.computeProjectionMatrixFOV(
            fov,
            self._res_x / self._res_y,
            near_plane,
            far_plane,
        )
        self._pose_reader = pose_reader
        self._debug = debug

    @property
    def state(self):
        position, orientation = self._pose_reader()

        eye, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (0, 0, 0.05),
            (1, 0, 0, 0)
        )

        to, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (0.1, 0, 0.05),
            (1, 0, 0, 0)
        )

        up, _ = self._pb_client.multiplyTransforms(
            position,
            orientation,
            (0, 0, 0.15),
            (1, 0, 0, 0)
        )

        view_matrix = self._pb_client.computeViewMatrix(
            eye,
            to,
            up,
        )

        _, _, rgb, depth_map, _ = self._pb_client.getCameraImage(
            width=self._res_x,
            height=self._res_y,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
            flags=pb.ER_NO_SEGMENTATION_MASK,
            viewMatrix=view_matrix,
            projectionMatrix=self._projection_matrix,
        )

        if self._debug:
            self._pb_client.addUserDebugLine(
                eye,
                to,
                (1, 0, 0),
                lifeTime=1.
            )

            self._pb_client.addUserDebugLine(
                eye,
                up,
                (0, 0, 1),
                lifeTime=1.
            )

        return rgb, depth_map
