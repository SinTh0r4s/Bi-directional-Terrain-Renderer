from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import moderngl
import numpy as np
from pyglm import glm

from .moderngl_util import get_uniform
from .shaders import load_program

if TYPE_CHECKING:
    from .camera import Camera
    from .terrain_picture import TerrainPicture


@dataclass
class TerrainPictureOverlayConfig:
    color_bias: glm.vec3 = field(default_factory=lambda: glm.vec3(1.0, 0.15, 0.45))
    blend_alpha: float = field(default=0.3)


class TerrainPictureOverlay:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._program = load_program(ctx, "image_overlay")
        # fmt: off
        self._vbo = ctx.buffer(
            np.array(
                [
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, -1.0, 1.0, 0.0,
                    -1.0, 1.0, 0.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                ], dtype=np.float32
            ).tobytes()
        )
        # fmt: on
        self._ibo = ctx.buffer(np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32).tobytes())
        self._vao = ctx._vertex_array(  # noqa: SLF001
            self._program,
            [(self._vbo, "2f 2f", "in_position", "in_uv")],
            index_buffer=self._ibo,
        )

    def __del__(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
        self._program.release()

    def render(self, camera: Camera, picture: TerrainPicture, config: TerrainPictureOverlayConfig) -> None:
        picture.bind_to_location(location=0)
        get_uniform(self._program, "image").value = 0
        get_uniform(self._program, "color_bias").write(config.color_bias.to_bytes())
        get_uniform(self._program, "blend_alpha").value = config.blend_alpha
        get_uniform(self._program, "image_over_viewport_aspect_ratio").value = picture.aspect_ratio / (
            camera.resolution.x / camera.resolution.y
        )
        get_uniform(self._program, "is_greyscale").value = picture.is_greyscale
        self._vao.render(mode=moderngl.TRIANGLES)
