from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl
import numpy as np
from pyglm import glm

from .moderngl_util import get_uniform
from .shaders import load_program

if TYPE_CHECKING:
    from .camera import Camera
    from .lighting import Lighting


class Skybox:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._program = load_program(ctx, "skybox")
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
        self._vao = ctx._vertex_array(  # noqa: SLF001
            self._program,
            [(self._vbo, "2f", "in_position")],
            index_buffer=self._ibo,
        )

    def __del__(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
        self._program.release()

    def render(self, camera: Camera, lighting: Lighting) -> None:
        get_uniform(self._program, "inv_proj").write(glm.inverse(camera.proj_matrix()).to_bytes())
        get_uniform(self._program, "inv_view").write(glm.inverse(camera.view_matrix()).to_bytes())
        get_uniform(self._program, "viewport_width_height").write(camera.resolution.to_bytes())
        get_uniform(self._program, "sky_zenith_color").write(lighting.sky_zenith_color.to_bytes())
        get_uniform(self._program, "sky_main_color").write(lighting.sky_main_color.to_bytes())
        get_uniform(self._program, "sky_horizon_color").write(lighting.sky_horizon_color.to_bytes())
        get_uniform(self._program, "sun_color").write(lighting.sun_color.to_bytes())
        get_uniform(self._program, "sun_direction").write(lighting.sun_direction.to_bytes())
        self._vao.render(mode=moderngl.TRIANGLES)
