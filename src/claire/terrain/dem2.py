import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import moderngl
import numpy as np

from claire.camera import Camera
from claire.terrain.lod_selector import MaxErrorLodSelector
from claire.terrain.numpy_types import HeightmapData
from claire.terrain.quadtree import QuadTree

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "vertex_shader2.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "fragment_shader.glsl"


def _create_index_buffer(size: int) -> np.ndarray:
    num_quads = (size - 1) * (size - 1)
    num_indices = num_quads * 6  # 2 triangles * 3 indices per quad

    index_data = np.empty(num_indices, dtype='u4')

    # Generate vertex corner indices for every quad cell
    r_indices, c_indices = np.mgrid[0:size - 1, 0:size - 1]
    top_left = (r_indices * size + c_indices).flatten()
    top_right = top_left + 1
    bottom_left = top_left + size
    bottom_right = bottom_left + 1

    index_data[0::6] = top_left
    index_data[1::6] = bottom_left
    index_data[2::6] = top_right
    index_data[3::6] = top_right
    index_data[4::6] = bottom_left
    index_data[5::6] = bottom_right

    return index_data


@dataclass
class TerrainTextures:
    texture_arrays_per_lod: dict[int, moderngl.TextureArray]
    tile_id_lookup: moderngl.Texture


def _ensure_array_size(array: HeightmapData, size: int) -> HeightmapData:
    cols, rows = array.shape
    return np.pad(array, ((0, size - cols), (0, size - rows)), "constant", constant_values=(0, 0))


def _upload_textures(ctx: moderngl.Context, heightmap: HeightmapData, tile_size: int, max_lod_level: int) -> TerrainTextures:
    cols, rows = heightmap.shape
    tile_cols, tile_rows = math.ceil(cols / tile_size), math.ceil(rows / tile_size)
    tiles: dict[int, list[HeightmapData]] = defaultdict(list)
    lookup_size = 1 << (max(tile_cols, tile_rows) + 1).bit_length()
    tile_id_lookup_array = np.full((lookup_size, lookup_size), -1, dtype=np.int32)
    sequential_id = 0
    for tile_col in range(tile_cols):
        for tile_row in range(tile_rows):
            col, row = tile_col * tile_size, tile_row * tile_size
            end_cols, end_rows = min(cols, col + tile_size), min(rows, row + tile_size)
            tile = heightmap[col:end_cols, row:end_rows]
            if np.max(tile) <= 0:
                continue
            tile_id_lookup_array[tile_col, tile_row] = sequential_id
            sequential_id += 1
            tiles[0].append(_ensure_array_size(tile, tile_size))
            for lod in range(1, max_lod_level + 1):
                stride = 1 << lod
                tiles[lod].append(_ensure_array_size(heightmap[col:end_cols:stride, row:end_rows: stride], tile_size >> lod))
    stacked_texture_data = {lod: np.stack(tiles[lod]) for lod in tiles}
    texture_arrays: dict[int, moderngl.TextureArray] = {}
    for lod, texture_data in stacked_texture_data.items():
        layers, cols, rows = texture_data.shape
        texture_arrays[lod] = ctx.texture_array(
            size=(cols, rows, layers),
            components=1,
            data=texture_data.tobytes(),
            dtype="f4",
        )
    tile_id_lookup_texture = ctx.texture(
        size=tile_id_lookup_array.shape,
        components=1,
        data=tile_id_lookup_array.tobytes(),
        dtype="i4"
    )
    return TerrainTextures(texture_arrays, tile_id_lookup_texture)


@dataclass
class Stats:
    duration_ms: float
    draw_calls: int
    vertices: int


class DEM:
    def __init__(self, ctx: moderngl.Context, heightmap: HeightmapData, max_y_error_in_px: float, lod_hystersis_factor: float = 0.1, max_lod_level: int = 5, mesh_size_exponent: int = 7, texture_tile_size_exponent: int = 10) -> None:
        if heightmap.dtype != np.float32:
            msg = "Requiring a heightmap of float32!"
            raise ValueError(msg)
        self._max_y_error_in_px = max_y_error_in_px
        self._lod_hysteresis_factor = lod_hystersis_factor
        self._max_lod_level = max_lod_level
        self._mesh_size_exponent = mesh_size_exponent
        self._texture_tile_size_exponent = texture_tile_size_exponent
        self._duration_s = 0

        self._quadtree = QuadTree(heightmap, mesh_size_exponent, max_lod_level)
        self._textures = _upload_textures(ctx, heightmap, 1 << self._texture_tile_size_exponent, max_lod_level)

        self._program = ctx.program(
            vertex_shader=self._bake_vertex_shader(_VERTEX_SHADER.read_text(encoding="utf-8")),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8")
        )

        self._mesh_size = (1 << mesh_size_exponent) + 1
        self._ibo = ctx.buffer(_create_index_buffer(self._mesh_size).tobytes())
        self._vao = ctx.vertex_array(self._program, [], index_buffer=self._ibo)

    def _bake_vertex_shader(self, raw_string: str) -> str:
        return raw_string.format(
            LOD_COUNT=self._max_lod_level + 1,
            MESH_SIZE_EXPONENT=self._mesh_size_exponent,
            TEXTURE_TILE_SIZE_EXPONENT=self._texture_tile_size_exponent,
        )

    def render(self, camera: Camera) -> None:
        self._draw_calls = 0
        time_start = time.time()
        for lod in range(self._max_lod_level + 1):
            self._textures.texture_arrays_per_lod[lod].use(location=lod)
        self._program["heightmap"].write(np.arange(self._max_lod_level + 1, dtype=np.uint32).tobytes())
        self._textures.tile_id_lookup.use(location=self._max_lod_level + 1)
        self._program["tile_id_lookup"] = self._max_lod_level + 1
        self._program["mvp"].write((camera.proj_matrix() * camera.view_matrix()).to_bytes())
        selection = self._quadtree.filter(MaxErrorLodSelector(camera, self._max_y_error_in_px, self._lod_hysteresis_factor))
        for chunk in selection:
            if not chunk.lod_data.aabb.is_visible(camera):
                continue
            self._program["lod_level"] = chunk.lod_level
            self._program["offset"].write(chunk.terrain_offset.to_bytes())
            # self._program["neighbor_lod_nwse"].write(selection.get_neighbor_lods_nwse(chunk).to_bytes())
            self._vao.render(mode=moderngl.TRIANGLES)
            self._draw_calls += 1
        self._duration_s = time.time() - time_start

    def stats(self) -> Stats:
        return Stats(self._duration_s * 1000, self._draw_calls, self._draw_calls * self._mesh_size)