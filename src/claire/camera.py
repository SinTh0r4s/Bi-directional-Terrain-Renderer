import math
from abc import ABC

from pyglm import glm


def _initial_yaw_and_pitch(camera_position: glm.vec3, look_at: glm.vec3) -> tuple[float, float]:
    direction = glm.normalize(look_at - camera_position)
    yaw = glm.degrees(glm.atan(direction.x, -direction.z))
    pitch = glm.degrees(glm.asin(direction.y))
    return yaw, pitch


class HasCameraPosition(ABC):
    position: glm.vec3


class HasViewProjMatrices(ABC):
    def view_matrix(self) -> glm.mat4: ...
    def proj_matrix(self) -> glm.mat4: ...


class Camera(HasCameraPosition, HasViewProjMatrices):
    def __init__(self, start_position: glm.vec3, look_at: glm.vec3):
        self.position = start_position
        self.yaw, self.pitch = _initial_yaw_and_pitch(start_position, look_at)

        self._move_speed = 10.0
        self._rotation_speed = 0.15
        self._slow_speed_factor = 1 / 25
        self._fast_speed_factor = 50

        self._left_active = False
        self._right_active = False
        self._up_active = False
        self._down_active = False
        self._forward_active = False
        self._backward_active = False
        self._fast_active = False
        self._slow_active = False

    def rotate(self, horizontal: float, vertical: float) -> None:
        self.yaw += horizontal * self._rotation_speed
        self.pitch += -vertical * self._rotation_speed
        self.pitch = glm.clamp(self.pitch, -89.0, 89.0)

    def start_left(self) -> None:
        self._left_active = True

    def stop_left(self) -> None:
        self._left_active = False

    def start_right(self) -> None:
        self._right_active = True

    def stop_right(self) -> None:
        self._right_active = False

    def start_up(self) -> None:
        self._up_active = True

    def stop_up(self) -> None:
        self._up_active = False

    def start_down(self) -> None:
        self._down_active = True

    def stop_down(self) -> None:
        self._down_active = False

    def start_forward(self) -> None:
        self._forward_active = True

    def stop_forward(self) -> None:
        self._forward_active = False

    def start_backward(self) -> None:
        self._backward_active = True

    def stop_backward(self) -> None:
        self._backward_active = False

    def start_fast(self) -> None:
        self._fast_active = True

    def stop_fast(self) -> None:
        self._fast_active = False

    def start_slow(self) -> None:
        self._slow_active = True

    def stop_slow(self) -> None:
        self._slow_active = False

    def on_update(self, delta_time: float) -> None:
        speed_multiplier = 1.0
        if self._slow_active:
            speed_multiplier *= self._slow_speed_factor
        if self._fast_active:
            speed_multiplier *= self._fast_speed_factor

        forward = self._forward_vector()
        right = self._right_vector()
        up = glm.cross(forward, right)

        if self._left_active:
            self.position += -right * self._move_speed * speed_multiplier * delta_time
        if self._right_active:
            self.position += right * self._move_speed * speed_multiplier * delta_time

        if self._up_active:
            self.position += up * self._move_speed * speed_multiplier * delta_time
        if self._down_active:
            self.position += -up * self._move_speed * speed_multiplier * delta_time

        if self._forward_active:
            self.position += forward * self._move_speed * speed_multiplier * delta_time
        if self._backward_active:
            self.position += -forward * self._move_speed * speed_multiplier * delta_time

    def _forward_vector(self) -> glm.vec3:
        yaw = glm.radians(self.yaw)
        pitch = glm.radians(self.pitch)
        return glm.normalize(glm.vec3(
            glm.sin(yaw) * glm.cos(pitch),
            glm.sin(pitch),
            -glm.cos(yaw) * glm.cos(pitch),
        ))

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
        return glm.perspective(math.radians(45.0), 4.0 / 3.0, 1.0, 25_000.0)
