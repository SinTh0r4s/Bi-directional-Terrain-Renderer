import math
import time
from typing import Optional

import pyglm.glm as glm

import moderngl
from PySide6.QtCore import QTimer, Qt, QPointF
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from claire.camera import Camera
from claire.dem import DEM, HEIGHTMAP

_KEY_LEFT = Qt.Key.Key_Left
_KEY_RIGHT = Qt.Key.Key_Right
_KEY_UP = Qt.Key.Key_Up
_KEY_DOWN = Qt.Key.Key_Down

class App(QOpenGLWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ctx: moderngl.Context
        self._dem = DEM(HEIGHTMAP)
        self._camera = Camera(start_position=glm.vec3(3.0, 1.0, 3.0), look_at=glm.vec3(0.0, 0.0, 0.0))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(30)  # ~30 FPS
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Would fire mouseMoveEvent even if no mouse button is down: self.setMouseTracking(True)
        self._last_mouse_pos: Optional[QPointF] = None
        self._left_mouse_button_pressed = False
        self._last_frame_time = time.time()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_mouse_button_pressed = True

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_mouse_button_pressed = False

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._last_mouse_pos is not None:
            dx = pos.x() - self._last_mouse_pos.x()
            dy = pos.y() - self._last_mouse_pos.y()
            self._camera.rotate(dx, dy)
        self._last_mouse_pos = pos

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._camera.forward(event.angleDelta().y())

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == _KEY_UP:
            self._camera.start_up()
        if event.key() == _KEY_DOWN:
            self._camera.start_down()
        if event.key() == _KEY_LEFT:
            self._camera.start_left()
        if event.key() == _KEY_RIGHT:
            self._camera.start_right()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == _KEY_UP:
            self._camera.stop_up()
        if event.key() == _KEY_DOWN:
            self._camera.stop_down()
        if event.key() == _KEY_LEFT:
            self._camera.stop_left()
        if event.key() == _KEY_RIGHT:
            self._camera.stop_right()

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        self._dem.bind(self._ctx)

    def paintGL(self):
        # Figure out which framebuffer is used by Qt for the widget and select it
        fbo = self._ctx.detect_framebuffer()
        fbo.use()

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.5, 0.5, 0.5)
        self._ctx.wireframe = True

        current_time = time.time()
        delta_time = current_time - self._last_frame_time
        self._last_frame_time = current_time

        self._camera.on_update(delta_time)
        view = self._camera.view_matrix()
        proj = glm.perspective(math.radians(45.0), 4.0 / 3.0, 0.1, 100.0)
        self._dem.render(proj * view)
