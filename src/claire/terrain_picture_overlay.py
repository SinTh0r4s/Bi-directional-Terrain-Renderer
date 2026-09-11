from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import moderngl
import numpy as np
from pyglm import glm

if TYPE_CHECKING:
    from claire.camera import Camera
    from claire.terrain_picture import TerrainPicture

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "image_overlay.vert.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "image_overlay.frag.glsl"


class TerrainPictureOverlay:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._program = ctx.program(
            vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8"),
        )
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
        self._vao = ctx.vertex_array(
            self._program,
            [(self._vbo, "2f 2f", "in_position", "in_uv")],
            index_buffer=self._ibo,
        )

    def __del__(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
        self._program.release()

    def render(self, camera: Camera, picture: TerrainPicture) -> None:
        picture.bind_to_location(location=0)
        self._program["image"] = 0
        color_bias = glm.vec3(1.0, 0.15, 0.45)
        self._program["color_bias"].write(color_bias.to_bytes())
        self._program["image_over_viewport_aspect_ratio"] = picture.aspect_ratio / (
            camera.resolution.x / camera.resolution.y
        )
        self._program["is_greyscale"] = picture.is_greyscale
        self._program["blend_alpha"] = 0.3
        self._vao.render(mode=moderngl.TRIANGLES)
