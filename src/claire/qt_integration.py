"""
ONLY USE THIS MODULE IF YOU ARE ALREADY USING QT!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .terrain_picture import TerrainPicture

if TYPE_CHECKING:
    import moderngl
    from PySide6.QtGui import QImage


def load_qt_image(ctx: moderngl.Context, image: QImage) -> TerrainPicture:
    return TerrainPicture(
        ctx, image.width(), image.height(), image.constBits(), "greyscale" if image.isGrayscale() else "color"
    )
