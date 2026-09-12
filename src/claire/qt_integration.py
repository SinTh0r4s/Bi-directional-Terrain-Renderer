"""
ONLY USE THIS MODULE IF YOU ARE ALREADY USING QT!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .terrain_picture import TerrainPicture

if TYPE_CHECKING:
    import moderngl
    from PyQt6.QtGui import QImage


def load_qt_image(ctx: moderngl.Context, image: QImage) -> TerrainPicture:
    # Do some legacy Qt magic
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    return TerrainPicture(ctx, image.width(), image.height(), ptr, "greyscale" if image.isGrayscale() else "color")
