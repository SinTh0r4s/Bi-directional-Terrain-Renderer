from typing import Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QKeyEvent
from pyglm import glm

from claire.camera import Camera

_KEY_LEFT = Qt.Key.Key_A
_KEY_RIGHT = Qt.Key.Key_D
_KEY_UP = Qt.Key.Key_Q
_KEY_DOWN = Qt.Key.Key_E
_KEY_FORWARD = Qt.Key.Key_W
_KEY_BACKWARD = Qt.Key.Key_S
_KEY_FAST_MODE = Qt.Key.Key_Control
_KEY_SLOW_MODE = Qt.Key.Key_Shift

class CameraControl:
    def __init__(self, position: glm.vec3, rotation: glm.vec3) -> None:
        self.camera = Camera(position, rotation)
        self._last_mouse_pos: Optional[QPointF] = None
        self._left_mouse_button_pressed = False

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

    def resizeGL(self, w: int, h: int) -> None:
        self.camera.resolution = glm.ivec2(w, h)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_mouse_button_pressed = True
            self._last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_mouse_button_pressed = False

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._last_mouse_pos is not None:
            d_yaw = pos.x() - self._last_mouse_pos.x()
            d_pitch = pos.y() - self._last_mouse_pos.y()
            self.camera.rotate(glm.vec3(0, -d_pitch, d_yaw))
        self._last_mouse_pos = pos

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == _KEY_UP:
            self._up_active = True
        if event.key() == _KEY_DOWN:
            self._down_active = True
        if event.key() == _KEY_LEFT:
            self._left_active = True
        if event.key() == _KEY_RIGHT:
            self._right_active = True
        if event.key() == _KEY_FORWARD:
            self._forward_active = True
        if event.key() == _KEY_BACKWARD:
            self._backward_active = True
        if event.key() == _KEY_FAST_MODE:
            self._fast_active = True
        if event.key() == _KEY_SLOW_MODE:
            self._slow_active = True

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == _KEY_UP:
            self._up_active = False
        if event.key() == _KEY_DOWN:
            self._down_active = False
        if event.key() == _KEY_LEFT:
            self._left_active = False
        if event.key() == _KEY_RIGHT:
            self._right_active = False
        if event.key() == _KEY_FORWARD:
            self._forward_active = False
        if event.key() == _KEY_BACKWARD:
            self._backward_active = False
        if event.key() == _KEY_FAST_MODE:
            self._fast_active = False
        if event.key() == _KEY_SLOW_MODE:
            self._slow_active = False

    def update(self, delta_time_sec: float) -> None:
        right = 0
        if self._left_active:
            right -= self._move_speed
        if self._right_active:
            right += self._move_speed

        up = 0
        if self._up_active:
            up += self._move_speed
        if self._down_active:
            up -= self._move_speed

        forward = 0
        if self._forward_active:
            forward += self._move_speed
        if self._backward_active:
            forward -= self._move_speed

        speed_multiplier = 1.0
        if self._slow_active:
            speed_multiplier *= self._slow_speed_factor
        if self._fast_active:
            speed_multiplier *= self._fast_speed_factor

        self.camera.translate_relative(
            glm.vec3(forward, right, up) * speed_multiplier * delta_time_sec
        )