from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import moderngl
import numpy as np
from pyglm import glm

from .moderngl_util import get_uniform

if TYPE_CHECKING:
    from .camera import Camera
    from .terrain_picture import TerrainPicture

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "image_overlay.vert.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "image_overlay.frag.glsl"


@dataclass
class TerrainPictureOverlayConfig:
    color_bias: glm.vec3 = field(default_factory=lambda: glm.vec3(1.0, 0.15, 0.45))
    blend_alpha: float = field(default=0.3)


class TerrainPictureOverlay:
    def __init__(self, ctx: moderngl.Context, config: TerrainPictureOverlayConfig) -> None:
        self._config = config
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

    def render(self, camera: Camera, picture: TerrainPicture) -> None:
        picture.bind_to_location(location=0)
        get_uniform(self._program, "image").value = 0
        get_uniform(self._program, "color_bias").write(self._config.color_bias.to_bytes())
        get_uniform(self._program, "blend_alpha").value = self._config.blend_alpha
        get_uniform(self._program, "image_over_viewport_aspect_ratio").value = picture.aspect_ratio / (
            camera.resolution.x / camera.resolution.y
        )
        get_uniform(self._program, "is_greyscale").value = picture.is_greyscale
        self._vao.render(mode=moderngl.TRIANGLES)
