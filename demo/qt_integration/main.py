import sys
import time

import pyglm.glm as glm

import moderngl
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QMainWindow, QApplication

from claire.aabb_renderer import AabbRenderer
from claire.skybox import Skybox
from claire.terrain.dem2 import DEM
from demo.heightmap_provider import load_heightmap
from demo.qt_integration.camera_control import CameraControl


class App(QOpenGLWidget):
    _ctx: moderngl.Context
    _dem: DEM
    _aabb_renderer: AabbRenderer
    _skybox: Skybox

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._camera_control = CameraControl(position=glm.vec3(6945, 3320, 8110), rotation=glm.vec3(0, -30, 164))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(30)  # ~30 FPS
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Would fire mouseMoveEvent even if no mouse button is down: self.setMouseTracking(True)
        self._last_frame_time = time.time()

    def mousePressEvent(self, event: QMouseEvent):
        self._camera_control.mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._camera_control.mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self._camera_control.mouseMoveEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        self._camera_control.keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        self._camera_control.keyReleaseEvent(event)

    def resizeGL(self, w: int, h: int, /) -> None:
        self._camera_control.resizeGL(w, h)

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        heightmap = load_heightmap()
        try:
            self._dem = DEM(self._ctx, heightmap, max_y_error_in_px=1.5)
            self._aabb_renderer = AabbRenderer(self._ctx)
            self._skybox = Skybox(self._ctx)
        except Exception as e:
            print(e)
            raise

    def paintGL(self):
        # Figure out which framebuffer is used by Qt for the widget and select it
        fbo = self._ctx.detect_framebuffer()
        fbo.use()

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.5, 0.5, 0.5)

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._skybox.render(self._camera_control.camera)
        self._ctx.enable(moderngl.DEPTH_TEST)

        current_time = time.time()
        delta_time = current_time - self._last_frame_time
        self._last_frame_time = current_time

        self._camera_control.update(delta_time)
        try:
            self._dem.render(self._camera_control.camera)
            self._aabb_renderer.render(self._camera_control.camera, self._dem.get_aabbs())
        except Exception as e:
            print(e)
            raise



def main() -> None:
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setCentralWidget(App())
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
