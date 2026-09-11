from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import moderngl
import numpy as np
from pyglm import glm

from claire.moderngl_util import get_uniform

if TYPE_CHECKING:
    from claire.aabb import AABB
    from claire.camera import Camera

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "aabb.vert.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "shaders" / "aabb.frag.glsl"


class AabbRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._program = ctx.program(
            vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8"),
        )
        # fmt: off
        self._vbo = ctx.buffer(
            np.array(
                [
                    # Front
                    0.0, 0.0, 1.0,
                    1.0, 0.0, 1.0,
                    1.0, 1.0, 1.0,
                    0.0, 1.0, 1.0,
                    # Back
                    0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0,
                    1.0, 1.0, 0.0,
                    0.0, 1.0, 0.0,
                ], dtype=np.float32
            ).tobytes()
        )
        self._ibo = ctx.buffer(
            np.array(
                [
                    0, 1, 1, 2, 2, 3, 3, 0,
                    4, 5, 5, 6, 6, 7, 7, 4,
                    0, 4, 1, 5, 2, 6, 3, 7,
                ], dtype=np.uint32
            ).tobytes()
        )
        # fmt: on
        self._vao = ctx._vertex_array(  # noqa: SLF001
            self._program,
            [(self._vbo, "3f", "in_position")],
            index_buffer=self._ibo,
        )
        self.color = glm.vec3(1.0, 1.0, 1.0)

    def __del__(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
        self._program.release()

    def render(self, camera: Camera, aabbs: list[AABB]) -> None:
        for aabb in aabbs:
            mvp = camera.proj_matrix() * camera.view_matrix() * aabb.get_unit_cube_model_matrix()
            get_uniform(self._program, "mvp").write(mvp.to_bytes())
            get_uniform(self._program, "color").write(self.color.to_bytes())
            self._vao.render(mode=moderngl.LINES)
