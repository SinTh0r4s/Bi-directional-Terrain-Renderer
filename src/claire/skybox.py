from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import moderngl
import numpy as np
from pyglm import glm

if TYPE_CHECKING:
    from claire.camera import Camera
    from claire.lighting import Lighting

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "skybox.vert.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "skybox.frag.glsl"


class Skybox:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._program = ctx.program(
            vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8"),
        )
        # fmt: off
        self._vbo = ctx.buffer(
            np.array(
                [
                    -1.0, -1.0,
                    1.0, -1.0,
                    -1.0, 1.0,
                    1.0, 1.0,
                ], dtype=np.float32
            ).tobytes()
        )
        # fmt: on
        self._ibo = ctx.buffer(np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32).tobytes())
        self._vao = ctx.vertex_array(
            self._program,
            [(self._vbo, "2f", "in_position")],
            index_buffer=self._ibo,
        )

    def render(self, camera: Camera, lighting: Lighting) -> None:
        self._program["inv_proj"].write(glm.inverse(camera.proj_matrix()).to_bytes())
        self._program["inv_view"].write(glm.inverse(camera.view_matrix()).to_bytes())
        self._program["viewport_width_height"].write(camera.resolution.to_bytes())
        self._program["sky_zenith_color"].write(lighting.sky_zenith_color.to_bytes())
        self._program["sky_main_color"].write(lighting.sky_main_color.to_bytes())
        self._program["sky_horizon_color"].write(lighting.sky_horizon_color.to_bytes())
        self._program["sun_color"].write(lighting.sun_color.to_bytes())
        self._program["sun_direction"].write(lighting.sun_direction.to_bytes())
        self._vao.render(mode=moderngl.TRIANGLES)
