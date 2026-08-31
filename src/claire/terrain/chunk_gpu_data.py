from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import moderngl
import numpy as np
from pyglm import glm

from claire.aabb import AABB
from claire.terrain.chunk_cpu_data import ChunkCpuData


@dataclass
class ChunkLodGpuData:
    size_exponent: int  # starting at 7 up to 10
    subchunks: list[list[Optional[AABB]]]

    @property
    def size(self) -> int:
        return 1 << self.size_exponent

@dataclass
class ChunkGpuData:
    chunk_sequential_id: int
    chunk_coords: glm.ivec2
    lods_per_exponent: dict[int, ChunkLodGpuData]


@dataclass
class GpuTerrain:
    chunks: list[ChunkGpuData]
    textures_per_lod: dict[int, moderngl.TextureArray]



def transfer_chunks_to_gpu(ctx: moderngl.Context, cpu_chunks: list[ChunkCpuData]) -> GpuTerrain:
    texture_lists: dict[int, list[np.ndarray]] = defaultdict(list)
    for chunk in cpu_chunks:
        for exponent, lod in chunk.lods_per_exponent.items():
            lod_size = 1 << lod.size_exponent
            shape = lod_size, lod_size
            if lod.heightmap.shape == shape:
                texture_lists[exponent].append(lod.heightmap)
            else:
                cols, rows = lod.heightmap.shape
                texture_lists[exponent].append(
                    np.pad(lod.heightmap, ((0, lod_size - cols), (0, lod_size - rows)), "constant", constant_values=(0, 0))
                )
    stacked_texture_data = {exponent: np.stack(texture_lists[exponent]) for exponent in texture_lists}
    texture_arrays: dict[int, moderngl.TextureArray] = {}
    for exponent, texture_data in stacked_texture_data.items():
        layers, cols, rows = texture_data.shape
        texture_arrays[exponent] = ctx.texture_array(
            size=(cols, rows, layers),
            components=1,
            data=texture_data.tobytes(),
            dtype="f4",
        )
    return GpuTerrain(
        chunks = [
            ChunkGpuData(
                chunk.chunk_sequential_id,
                chunk.chunk_coords,
                {
                    exponent: ChunkLodGpuData(lod.size_exponent, lod.subchunks)
                    for exponent, lod in chunk.lods_per_exponent.items()
                }
            )
            for chunk in cpu_chunks
        ],
        textures_per_lod = texture_arrays,
    )