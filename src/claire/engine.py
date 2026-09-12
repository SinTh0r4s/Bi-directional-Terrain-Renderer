from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import moderngl

from .aabb_renderer import AabbRenderer
from .skybox import Skybox
from .terrain.dem import DEM, DemConfig
from .terrain_picture import TerrainPicture
from .terrain_picture_overlay import TerrainPictureOverlay

if TYPE_CHECKING:

    from .camera import Camera
    from .config import Config
    from .terrain.numpy_types import HeightmapData


class Engine:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._dem: DEM | None = None
        self._pictures: list[TerrainPicture] = []
        self._active_picture: TerrainPicture | None = None
        self._ctx = moderngl.create_context()
        self._ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self._image_overlay = TerrainPictureOverlay(self._ctx)
        self._aabb_renderer = AabbRenderer(self._ctx)
        self._skybox = Skybox(self._ctx)

    def load_picture(
        self,
        width: int,
        height: int,
        image: bytes | bytearray | memoryview,
        image_format: Literal["greyscale", "color"],
    ) -> TerrainPicture:
        picture = TerrainPicture(self._ctx, width, height, image, image_format)
        self._pictures.append(picture)
        return picture

    def set_overlay_picture(self, picture: TerrainPicture) -> None:
        self._active_picture = picture

    def remove_overlay_picture(self) -> None:
        self._active_picture = None

    def load_heightmap(self, heightmap: HeightmapData, config: DemConfig) -> None:
        self._dem = DEM(self._ctx, heightmap, config, self._config.lod_config)

    def draw(self, camera: Camera) -> None:
        # Figure out which framebuffer is used by Qt for the widget and select it
        fbo = self._ctx.detect_framebuffer()
        fbo.use()

        self._ctx.enable(moderngl.CULL_FACE)
        self._ctx.clear(0.5, 0.5, 0.5)

        self._skybox.render(camera, self._config.lighting)

        if self._dem is not None:
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._dem.render(camera, self._config.lighting)
            if self._config.print_stats is not None:
                self._config.print_stats(self._dem.stats())
            if self._config.show_terrain_bounding_boxes:
                self._aabb_renderer.render(camera, self._dem.get_aabbs())
            self._ctx.disable(moderngl.DEPTH_TEST)

        if self._active_picture is not None:
            self._ctx.enable(moderngl.BLEND)
            self._image_overlay.render(camera, self._active_picture, self._config.picture_overlay)
            self._ctx.disable(moderngl.BLEND)
