from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pyglm import glm


class HasCameraPosition(ABC):
    position: glm.vec3


class HasViewProjMatrices(ABC):
    @abstractmethod
    def view_matrix(self) -> glm.mat4: ...
    @abstractmethod
    def proj_matrix(self) -> glm.mat4: ...


class HasCameraPositionResolutionFovNearplane(HasCameraPosition, ABC):
    resolution: glm.ivec2
    fov_deg: float
    near_plane: float


class Camera(HasCameraPositionResolutionFovNearplane, HasViewProjMatrices):
    def __init__(self, position: glm.vec3, rotation: glm.vec3) -> None:
        self.position = position
        self.rotation = rotation
        self.fov_deg = 45
        self.resolution = glm.ivec2(800, 600)
        self.near_plane = 1
        self.far_plane = 25_000

    def copy(self) -> Camera:
        new_camera = Camera(self.position, self.rotation)
        new_camera.fov_deg = self.fov_deg
        new_camera.resolution = self.resolution
        new_camera.near_plane = self.near_plane
        new_camera.far_plane = self.far_plane
        return new_camera

    def look_at(self, target: glm.vec3) -> None:
        direction = glm.normalize(target - self.position)
        self.rotation = glm.vec3(
            0, glm.degrees(glm.asin(direction.y)), glm.degrees(glm.atan(direction.x, -direction.z))
        )

    def rotate(self, offset: glm.vec3) -> None:
        self.rotation += offset
        self.rotation.y = glm.clamp(self.rotation.y, -89.0, 89.0)

    def translate_relative(self, forward_right_up: glm.vec3) -> None:
        forward = self._forward_vector()
        right = self._right_vector()
        up = glm.cross(forward, right)
        self.position += forward * forward_right_up.x + right * forward_right_up.y + up * forward_right_up.z

    def _forward_vector(self) -> glm.vec3:
        yaw = glm.radians(self.rotation.z)
        pitch = glm.radians(self.rotation.y)
        return glm.normalize(
            glm.vec3(
                glm.sin(yaw) * glm.cos(pitch),
                glm.sin(pitch),
                -glm.cos(yaw) * glm.cos(pitch),
            )
        )

    def _right_vector(self) -> glm.vec3:
        return glm.normalize(
            glm.cross(
                self._forward_vector(),
                glm.vec3(0.0, 1.0, 0.0),
            )
        )

    def view_matrix(self) -> glm.mat4:
        return glm.lookAt(
            self.position,
            self.position + self._forward_vector(),
            glm.vec3(0.0, 1.0, 0.0),
        )

    def proj_matrix(self) -> glm.mat4:
        return glm.perspective(
            math.radians(self.fov_deg), self.resolution.x / self.resolution.y, self.near_plane, self.far_plane
        )
