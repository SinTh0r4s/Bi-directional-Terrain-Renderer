from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import moderngl
from pyglm import glm
from PySide6.QtCore import Qt, QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow
from typing_extensions import override

from claire.aabb_renderer import AabbRenderer
from claire.lighting import Lighting
from claire.skybox import Skybox
from claire.terrain.dem import DEM
from demo.heightmap_provider import load_heightmap
from demo.qt_integration.camera_control import CameraControl

if TYPE_CHECKING:
    from PySide6.QtGui import QKeyEvent, QMouseEvent


class App(QOpenGLWidget):
    _ctx: moderngl.Context
    _dem: DEM
    _aabb_renderer: AabbRenderer
    _skybox: Skybox

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._lighting = Lighting()
        self._camera_control = CameraControl(position=glm.vec3(6945, 3320, 8110), rotation=glm.vec3(0, -30, 164))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(30)  # ~30 FPS
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Would fire mouseMoveEvent even if no mouse button is down: self.setMouseTracking(True)
        self._last_frame_time = time.time()

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._camera_control.mouse_press_event(event)

    @override
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._camera_control.mouse_release_event(event)

    @override
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._camera_control.mouse_move_event(event)

    @override
    def keyPressEvent(self, event: QKeyEvent) -> None:
        self._camera_control.key_press_event(event)

    @override
    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self._camera_control.key_release_event(event)

    @override
    def resizeGL(self, w: int, h: int, /) -> None:
        self._camera_control.resize_gl(w, h)

    @override
    def initializeGL(self) -> None:
        self._ctx = moderngl.create_context()
        heightmap = load_heightmap()
        try:
            self._dem = DEM(self._ctx, heightmap, max_y_error_in_px=1.5)
            self._aabb_renderer = AabbRenderer(self._ctx)
            self._skybox = Skybox(self._ctx)
        except Exception as e:
            print(e)  # noqa: T201
            raise

    @override
    def paintGL(self) -> None:
        # Figure out which framebuffer is used by Qt for the widget and select it
        fbo = self._ctx.detect_framebuffer()
        fbo.use()

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.5, 0.5, 0.5)

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._skybox.render(self._camera_control.camera, self._lighting)
        self._ctx.enable(moderngl.DEPTH_TEST)

        current_time = time.time()
        delta_time = current_time - self._last_frame_time
        self._last_frame_time = current_time

        self._camera_control.update(delta_time)
        try:
            self._dem.render(self._camera_control.camera, self._lighting)
            print(self._dem.stats())
            self._aabb_renderer.render(self._camera_control.camera, self._dem.get_aabbs())
        except Exception as e:
            print(e)  # noqa: T201
            raise


def main() -> None:
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setCentralWidget(App())
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
