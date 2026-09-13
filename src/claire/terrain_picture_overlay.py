from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyglm import glm

from .fullscreen_quad import FullscreenQuad

if TYPE_CHECKING:
    import moderngl

    from .camera import Camera
    from .terrain_picture import TerrainPicture


@dataclass
class TerrainPictureOverlayConfig:
    color_bias: glm.vec3 = field(default_factory=lambda: glm.vec3(1.0, 0.15, 0.45))
    blend_alpha: float = field(default=0.3)


class TerrainPictureOverlay:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._quad = FullscreenQuad(ctx, "image_overlay")

    def render(self, camera: Camera, picture: TerrainPicture, config: TerrainPictureOverlayConfig) -> None:
        picture.bind_to_location(location=0)
        self._quad.get_uniform("image").value = 0
        self._quad.get_uniform("color_bias").write(config.color_bias.to_bytes())
        self._quad.get_uniform("blend_alpha").value = config.blend_alpha
        self._quad.get_uniform("image_over_viewport_aspect_ratio").value = picture.aspect_ratio / camera.aspect_ratio
        self._quad.get_uniform("is_greyscale").value = picture.is_greyscale
        self._quad.render()
