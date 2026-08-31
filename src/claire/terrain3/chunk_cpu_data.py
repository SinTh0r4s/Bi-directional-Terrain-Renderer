import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyglm import glm

from claire.aabb import AABB


@dataclass
class ChunkLodCpuData:
    size_exponent: int  # starting at 7 up to 10
    heightmap: np.ndarray
    subchunks: list[list[Optional[AABB]]]

    @property
    def size(self) -> int:
        return 1 << self.size_exponent

@dataclass
class ChunkCpuData:
    chunk_sequential_id: int
    chunk_coords: glm.ivec2
    lods_per_exponent: dict[int, ChunkLodCpuData]


def _load_lod(chunk_coords: glm.ivec2, heightmap_chunk: np.ndarray, size_exponent: int,  chunk_size_exponent: int, subchunk_size_exponent: int) -> ChunkLodCpuData:
    chunk_size = 1 << chunk_size_exponent
    subchunk_size = 1 << subchunk_size_exponent
    in_world_chunk_coords = chunk_coords * chunk_size
    size = 1 << size_exponent
    stride = 1 << (chunk_size_exponent - size_exponent)
    heightmap_chunk_lod = heightmap_chunk[::stride, ::stride]
    cols, rows = heightmap_chunk_lod.shape
    subchunks: list[list[Optional[AABB]]] = []
    for col in range(math.ceil(size / subchunk_size)):
        subchunks_col: list[Optional[AABB]] = []
        for row in range(math.ceil(size / subchunk_size)):
            heightmap_subchunk = heightmap_chunk_lod[col: min((col + 1) * subchunk_size, cols), row: min((row + 1) * subchunk_size, rows)]
            min_value = max(0, np.min(heightmap_subchunk))
            max_value = np.max(heightmap_subchunk)
            if max_value <= 0:
                subchunks_col.append(None)
            else:
                in_world_coords = glm.ivec2(col * stride, row * stride) + in_world_chunk_coords
                position_min = glm.vec3(in_world_coords.x, min_value, in_world_coords.y)
                position_max = glm.vec3(in_world_coords.x + chunk_size, max_value, in_world_coords.y + chunk_size)
                subchunks_col.append(AABB(position_min, position_max))
        subchunks.append(subchunks_col)
    return ChunkLodCpuData(size_exponent, heightmap_chunk_lod, subchunks)


def _load_chunk(chunk_coords: glm.ivec2, chunk_sequential_id: int, heightmap_chunk: np.ndarray, chunk_size_exponent: int, subchunk_size_exponent: int) -> ChunkCpuData:
    exponents = list(range(subchunk_size_exponent, chunk_size_exponent + 1))
    return ChunkCpuData(
        chunk_sequential_id,
        chunk_coords,
        {
            size_exponent: _load_lod(chunk_coords, heightmap_chunk, size_exponent, chunk_size_exponent, subchunk_size_exponent)
            for size_exponent in exponents
        }
    )


def load_chunks_into_cpu(heightmap: np.ndarray, chunk_size_exponent: int, subchunk_size_exponent: int) -> list[ChunkCpuData]:
    if heightmap.dtype != np.float32:
        msg = "heightmap must be provided in float32"
        raise ValueError(msg)
    cols, rows = heightmap.shape
    stride = 1 << chunk_size_exponent
    chunks: list[ChunkCpuData] = []
    chunk_sequencial_id = 0
    for chunk_col in range(math.ceil(cols / stride)):
        for chunk_row in range(math.ceil(rows / stride)):
            col_start = chunk_col * stride
            col_end = min(col_start + stride, cols)
            row_start = chunk_row * stride
            row_end = min(row_start + stride, rows)
            chunk_data = heightmap[col_start: col_end, row_start: row_end]
            if np.max(chunk_data) > 0:
                chunks.append(_load_chunk(glm.ivec2(chunk_col, chunk_row), chunk_sequencial_id, chunk_data, chunk_size_exponent, subchunk_size_exponent))
                chunk_sequencial_id += 1
    return chunks


if __name__ == "__main__":
    from pathlib import Path
    from skimage.io import imread
    tif = Path(__file__).parent.parent / "DSM_1m_UTM11N.tif"
    # run with 1024² chunks and 128² subchunks (= mesh size)
    _ = load_chunks_into_cpu(imread(tif), 10, 7)
