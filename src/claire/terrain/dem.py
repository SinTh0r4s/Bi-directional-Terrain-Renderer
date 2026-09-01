import math

from claire.terrain.chunk import Chunk
from pathlib import Path
from typing import Final

import numpy as np
import moderngl
from skimage.io import imread

from claire.camera import Camera
from claire.terrain.chunk_cpu_data import load_chunks_into_cpu
from claire.terrain.chunk_gpu_data import transfer_chunks_to_gpu

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "vertex_shader.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "fragment_shader.glsl"


def _create_index_buffer(rows: float, cols: float) -> np.ndarray:
    num_quads = (rows - 1) * (cols - 1)
    num_indices = num_quads * 6  # 2 triangles * 3 indices per quad

    index_data = np.empty(num_indices, dtype='u4')

    # Generate vertex corner indices for every quad cell
    r_indices, c_indices = np.mgrid[0:rows - 1, 0:cols - 1]
    top_left = (r_indices * cols + c_indices).flatten()
    top_right = top_left + 1
    bottom_left = top_left + cols
    bottom_right = bottom_left + 1

    index_data[0::6] = top_left
    index_data[1::6] = bottom_left
    index_data[2::6] = top_right
    index_data[3::6] = top_right
    index_data[4::6] = bottom_left
    index_data[5::6] = bottom_right

    return index_data


class DEM:
    def __init__(self, ctx: moderngl.Context, heightmap_tif: Path, chunk_size = 1024, mesh_size: int = 128) -> None:
        """Heightmap values of zero or below are automatically discarded and not shown"""
        self._chunk_size_exponent = math.floor(math.log2(chunk_size))
        self._mesh_size_exponent = math.floor(math.log2(mesh_size))
        heightmap = imread(heightmap_tif)
        if heightmap.dtype != np.float32:
            msg = "Requiring a heightmap of float32!"
            raise ValueError(msg)
        heightmap[heightmap < 0] = 0
        self._terrain_data = transfer_chunks_to_gpu(ctx, load_chunks_into_cpu(heightmap, self._chunk_size_exponent, self._mesh_size_exponent))
        del heightmap  # free memory  TODO: validate!
        self._chunks = [Chunk(chunk_data, self._chunk_size_exponent) for chunk_data in self._terrain_data.chunks]

        self._program = ctx.program(
            vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8")
        )
        mesh_size = 1 << self._mesh_size_exponent
        clustered_indices = _create_index_buffer(mesh_size, mesh_size)
        self._ibo = ctx.buffer(clustered_indices.tobytes())
        del clustered_indices  # free memory
        self._vao = ctx.vertex_array(self._program, [], index_buffer=self._ibo)

    def render(self, camera: Camera) -> None:
        texture_location_per_exponent: dict[int, int] = {}
        current_location = 0
        for exponent, texture_array in self._terrain_data.textures_per_lod.items():
            texture_array.use(location=current_location)
            texture_location_per_exponent[exponent] = current_location
            current_location += 1

        for chunk in self._chunks:
            chunk.render(self._program, self._vao, camera, texture_location_per_exponent)
