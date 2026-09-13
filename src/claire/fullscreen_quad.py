from __future__ import annotations

import moderngl
import numpy as np

from .moderngl_util import get_uniform
from .shaders import BakeConstant, load_program


class FullscreenQuad:
    def __init__(self, ctx: moderngl.Context, frag: str | tuple[str, list[BakeConstant]]) -> None:
        self._program = load_program(ctx, "fullscreen_quad", frag)
        # fmt: off
        self._vbo = ctx.buffer(
            np.array(
                [
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, -1.0, 1.0, 0.0,
                    -1.0, 1.0, 0.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                ],
                dtype=np.float32,
            ).tobytes()
        )
        # fmt: on
        self._ibo = ctx.buffer(np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32).tobytes())
        self._vao = ctx._vertex_array(  # noqa: SLF001
            self._program,
            [(self._vbo, "2f 2f", "in_position", "in_uv")]
            if "in_uv" in self._program
            else [(self._vbo, "2f 2x4", "in_position")],
            index_buffer=self._ibo,
        )

    def __del__(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
        self._program.release()

    def get_uniform(self, label: str) -> moderngl.Uniform:
        return get_uniform(self._program, label)

    def render(self) -> None:
        self._vao.render(mode=moderngl.TRIANGLES)
