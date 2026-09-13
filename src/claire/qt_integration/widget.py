"""
ONLY USE THIS MODULE IF YOU ARE ALREADY USING QT!
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, override

from pyglm import glm
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QKeyEvent, QMouseEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ..engine import Engine
from ..terrain.dem import DemConfig
from .camera_control import CameraControl

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import Config
    from ..terrain.numpy_types import HeightmapData
    from ..terrain_picture import TerrainPicture


def _load_qt_image(engine: Engine, image: QImage) -> TerrainPicture:
    # Do some legacy Qt magic
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    data = bytes(ptr)  # pyright: ignore[reportArgumentType]
    return engine.load_picture(image.width(), image.height(), data, "greyscale" if image.isGrayscale() else "color")


class TerrainViewer(QOpenGLWidget):
    _engine: Engine
    _overlay_picture: TerrainPicture

    def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self._config = config
        self._camera_control = CameraControl(position=glm.vec3(6945, 3320, 8110), rotation=glm.vec3(0, -30, 164))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self._timer.start(30)  # ~30 FPS
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Would fire mouseMoveEvent even if no mouse button is down: self.setMouseTracking(True)
        self._last_frame_time = time.time()
        self._heightmap: HeightmapData | None = None
        self._picture: Path | None = None

    @override
    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None:
            self._camera_control.mouse_press_event(a0)

    @override
    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None:
            self._camera_control.mouse_release_event(a0)

    @override
    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None:
            self._camera_control.mouse_move_event(a0)

    @override
    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None:
            self._camera_control.key_press_event(a0)

    @override
    def keyReleaseEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None:
            self._camera_control.key_release_event(a0)

    @override
    def resizeGL(self, w: int, h: int) -> None:
        self._camera_control.resize_gl(w, h)

    @override
    def initializeGL(self) -> None:
        try:
            self._engine = Engine(self._config)
        except Exception as e:
            print(e)  # noqa: T201
            raise

    @override
    def paintGL(self) -> None:
        current_time = time.time()
        delta_time = current_time - self._last_frame_time
        self._last_frame_time = current_time
        self._camera_control.update(delta_time)
        try:
            if self._heightmap is not None:
                self._engine.load_heightmap(self._heightmap, DemConfig())
                self._heightmap = None
            if self._picture is not None:
                self._overlay_picture = _load_qt_image(self._engine, QImage(str(self._picture.resolve())))
                self._engine.set_overlay_picture(self._overlay_picture)
                self._picture = None
            self._engine.draw(self._camera_control.camera)
        except Exception as e:
            print(e)  # noqa: T201
            raise

    def load_heightmap(self, heightmap: HeightmapData) -> None:
        # cannot call engine directly. This MUST happen from inside a xxxGL() callback
        self._heightmap = heightmap

    def set_overlay_image(self, image: Path) -> None:
        self._picture = image
