import time
from pathlib import Path
from typing import Optional

import pyglm.glm as glm

import moderngl
from PySide6.QtCore import QTimer, Qt, QPointF
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from claire.camera import Camera
from claire.terrain3.dem import DEM

_KEY_LEFT = Qt.Key.Key_A
_KEY_RIGHT = Qt.Key.Key_D
_KEY_UP = Qt.Key.Key_Q
_KEY_DOWN = Qt.Key.Key_E
_KEY_FORWARD = Qt.Key.Key_W
_KEY_BACKWARD = Qt.Key.Key_S

class App(QOpenGLWidget):
    _ctx: moderngl.Context
    _dem: DEM

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
            self._last_mouse_pos = event.position()

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

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == _KEY_UP:
            self._camera.start_up()
        if event.key() == _KEY_DOWN:
            self._camera.start_down()
        if event.key() == _KEY_LEFT:
            self._camera.start_left()
        if event.key() == _KEY_RIGHT:
            self._camera.start_right()
        if event.key() == _KEY_FORWARD:
            self._camera.start_forward()
        if event.key() == _KEY_BACKWARD:
            self._camera.start_backward()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == _KEY_UP:
            self._camera.stop_up()
        if event.key() == _KEY_DOWN:
            self._camera.stop_down()
        if event.key() == _KEY_LEFT:
            self._camera.stop_left()
        if event.key() == _KEY_RIGHT:
            self._camera.stop_right()
        if event.key() == _KEY_FORWARD:
            self._camera.stop_forward()
        if event.key() == _KEY_BACKWARD:
            self._camera.stop_backward()

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        try:
            self._dem = DEM(self._ctx, Path(__file__).parent / "DSM_1m_UTM11N.tif")
        except Exception as e:
            print(e)
            raise

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
        try:
            self._dem.render(self._camera)
        except Exception as e:
            print(e)
            raise
