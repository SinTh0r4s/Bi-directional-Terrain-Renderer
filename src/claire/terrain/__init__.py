from pathlib import Path
from typing import Final

import moderngl

from claire.camera import Camera
from claire.terrain.chunk import Chunk
from claire.terrain.load_data import load_geodata, DataChunk

__all__ = ["Terrain"]

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "vertex_shader.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "fragment_shader.glsl"


class Terrain:
    def __init__(self, tif: Path) -> None:
        self._chunks = [Chunk(data_chunk) for data_chunk in load_geodata(tif)]

    def upload(self, ctx: moderngl.Context) -> None:
        self._program = ctx.program(
            vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8")
        )
        for chunk in self._chunks:
            chunk.upload(ctx, self._program)

    def render(self, camera: Camera) -> None:
        for chunk in self._chunks:
            chunk.render(camera, self._program)