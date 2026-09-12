from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import moderngl
import numpy as np

from ..moderngl_util import get_uniform
from ..terrain.lod_selector import CullingLodSelector, LodConfig
from ..terrain.quadtree import QuadTree

if TYPE_CHECKING:
    from ..aabb import AABB
    from ..camera import Camera
    from ..lighting import Lighting
    from ..terrain.numpy_types import HeightmapData

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "terrain.vert.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "terrain.frag.glsl"


@dataclass(frozen=True)
class DemConfig:
    """There values can only be set once when the DEM class is initialized."""

    max_lod_level: int = 5
    mesh_size_exponent: int = 7
    texture_tile_size_exponent: int = 10


def _create_index_buffer(size: int) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    num_quads = (size - 1) * (size - 1)
    num_indices = num_quads * 6  # 2 triangles * 3 indices per quad

    index_data = np.empty(num_indices, dtype=np.uint32)

    # Generate vertex corner indices for every quad cell
    r_indices, c_indices = np.mgrid[0 : size - 1, 0 : size - 1]
    top_left = (r_indices * size + c_indices).flatten()
    top_right = top_left + 1
    bottom_left = top_left + size
    bottom_right = bottom_left + 1

    index_data[0::6] = top_left
    index_data[1::6] = top_right
    index_data[2::6] = bottom_left
    index_data[3::6] = top_right
    index_data[4::6] = bottom_right
    index_data[5::6] = bottom_left

    return index_data


@dataclass
class TerrainTextures:
    texture_arrays_per_lod: dict[int, moderngl.TextureArray]
    tile_id_lookup: moderngl.Texture


def _ensure_array_size(array: HeightmapData, size: int) -> HeightmapData:
    cols, rows = array.shape
    return np.pad(array, ((0, size - cols), (0, size - rows)), "constant", constant_values=(0, 0))


def _upload_textures(
    ctx: moderngl.Context, heightmap: HeightmapData, tile_size: int, max_lod_level: int
) -> TerrainTextures:
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
                tiles[lod].append(
                    _ensure_array_size(heightmap[col:end_cols:stride, row:end_rows:stride], tile_size >> lod)
                )
    stacked_texture_data = {lod: np.stack(tiles[lod]) for lod in tiles}
    texture_arrays: dict[int, moderngl.TextureArray] = {}
    for lod, texture_data in stacked_texture_data.items():
        layers, cols, rows = texture_data.shape
        texture_arrays[lod] = ctx.texture_array(
            size=(cols, rows, layers),
            components=1,
            data=np.ascontiguousarray(texture_data.transpose(0, 2, 1)).tobytes(),
            dtype="f4",
        )
    tile_id_lookup_texture = ctx.texture(
        size=cast("tuple[int, int]", tile_id_lookup_array.shape),
        components=1,
        data=np.ascontiguousarray(tile_id_lookup_array.T).tobytes(),
        dtype="i4",
    )
    return TerrainTextures(texture_arrays, tile_id_lookup_texture)


@dataclass
class Stats:
    duration_ms: float
    draw_calls: int
    vertices: int

    def __str__(self) -> str:
        return f"{round(self.duration_ms)}ms    {self.draw_calls} calls    {self.vertices:_} vertices"


class DEM:
    def __init__(
        self, ctx: moderngl.Context, heightmap: HeightmapData, config: DemConfig, lod_config: LodConfig
    ) -> None:
        if heightmap.dtype != np.float32:
            msg = "Requiring a heightmap of float32!"
            raise ValueError(msg)
        self._config = config
        self._lod_config = lod_config
        self._duration_s = 0

        self._quadtree = QuadTree(heightmap, self._config.mesh_size_exponent, self._config.max_lod_level)
        self._textures = _upload_textures(
            ctx,
            heightmap,
            1 << self._config.texture_tile_size_exponent,
            self._config.max_lod_level,
        )

        self._program = ctx.program(
            vertex_shader=self._bake_vertex_shader(_VERTEX_SHADER.read_text(encoding="utf-8")),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8"),
        )

        self._mesh_size = (1 << self._config.mesh_size_exponent) + 1
        self._ibo = ctx.buffer(_create_index_buffer(self._mesh_size).tobytes())
        self._vao = ctx._vertex_array(self._program, [], index_buffer=self._ibo)  # noqa: SLF001

    def __del__(self) -> None:
        self._vao.release()
        self._ibo.release()
        self._program.release()
        self._textures.tile_id_lookup.release()
        for texture_array in self._textures.texture_arrays_per_lod.values():
            texture_array.release()

    def _bake_vertex_shader(self, raw_string: str) -> str:
        return raw_string.format(
            LOD_COUNT=self._config.max_lod_level + 1,
            MESH_SIZE_EXPONENT=self._config.mesh_size_exponent,
            TEXTURE_TILE_SIZE_EXPONENT=self._config.texture_tile_size_exponent,
        )

    def render(self, camera: Camera, lighting: Lighting) -> None:
        self._draw_calls = 0
        time_start = time.time()
        for lod in range(self._config.max_lod_level + 1):
            self._textures.texture_arrays_per_lod[lod].use(location=lod)
        get_uniform(self._program, "heightmap").write(
            np.arange(self._config.max_lod_level + 1, dtype=np.uint32).tobytes()
        )
        self._textures.tile_id_lookup.use(location=self._config.max_lod_level + 1)
        get_uniform(self._program, "tile_id_lookup").value = self._config.max_lod_level + 1
        get_uniform(self._program, "mvp").write((camera.proj_matrix() * camera.view_matrix()).to_bytes())
        get_uniform(self._program, "camera_position").write(camera.position.to_bytes())
        get_uniform(self._program, "terrain_default_color").write(lighting.terrain_default_color.to_bytes())
        get_uniform(self._program, "sun_direction").write(lighting.sun_direction.to_bytes())
        get_uniform(self._program, "sun_color").write(lighting.sun_color.to_bytes())
        selection = self._quadtree.filter(CullingLodSelector(camera, self._lod_config))
        for chunk in selection:
            if not chunk.lod_data.aabb.is_visible(camera):
                continue
            get_uniform(self._program, "lod_level").value = chunk.lod_level
            get_uniform(self._program, "offset").write(chunk.terrain_offset.to_bytes())
            get_uniform(self._program, "neighbor_lod_nwse").write(selection.get_neighbor_lods_nwse(chunk).to_bytes())
            self._vao.render(mode=moderngl.TRIANGLES)
            self._draw_calls += 1
        self._duration_s = time.time() - time_start

    def stats(self) -> Stats:
        return Stats(self._duration_s * 1000, self._draw_calls, self._draw_calls * self._mesh_size * self._mesh_size)

    def get_aabbs(self) -> list[AABB]:
        return self._quadtree.get_aabbs()
