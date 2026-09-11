from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import numpy as np
from pyglm import glm

from claire.aabb import AABB

if TYPE_CHECKING:
    from claire.camera import Camera
    from claire.terrain.numpy_types import HeightmapData


@dataclass
class LodData:
    aabb: AABB
    lod_level: int


_T = TypeVar("_T")


class LodSelector(ABC, Generic[_T]):
    def should_refine(self, lod_data: _T) -> Literal["cull", "refine", "render", "use_previous"]: ...


class CullingLodSelector(LodSelector[LodData]):
    def __init__(self, camera: Camera, max_error_in_px: float, hysteresis_factor: float = 0.1) -> None:
        self._camera = camera
        self._max_error_in_px = max_error_in_px
        self._hysteresis_factor = hysteresis_factor

    def update_camera(self, camera: Camera) -> None:
        self._camera = camera

    def should_refine(self, lod_data: LodData) -> Literal["cull", "refine", "render", "use_previous"]:
        if not lod_data.aabb.is_visible(self._camera):
            return "cull"
        distance = lod_data.aabb.get_shortest_distance_to(self._camera)
        base_distance = 30
        effective_distance = base_distance * (1 << lod_data.lod_level) - 100
        if distance >= effective_distance * (1 - self._hysteresis_factor):
            return "render"
        if distance <= effective_distance:
            return "refine"
        return "use_previous"


def extract_lod_aabb(full_heightmap: HeightmapData, offset: glm.ivec2, lod_stride: int, lod_width: int) -> AABB | None:
    full_height_size = lod_stride * lod_width
    full_cols, full_rows = full_heightmap.shape
    max_col = min(full_cols, offset.x + full_height_size)
    max_row = min(full_rows, offset.y + full_height_size)
    lod_heightmap = full_heightmap[offset.x : max_col, offset.y : max_row]
    if lod_heightmap.shape[0] == 0 or lod_heightmap.shape[1] == 0:
        return None
    max_height = np.max(lod_heightmap)
    if max_height <= 0:
        return None
    min_height = np.min(lod_heightmap[lod_heightmap > 0])
    return AABB(glm.vec3(offset.x, min_height, offset.y), glm.vec3(max_col, max_height, max_row))


def create_lod_data(
    heightmap: HeightmapData, terrain_offset: glm.ivec2, lod_stride: int, lod_size: int
) -> LodData | None:
    aabb = extract_lod_aabb(heightmap, terrain_offset, lod_stride, lod_size)
    if aabb is None:
        return None
    return LodData(aabb, int(math.log2(lod_stride)))  # , max_error_y)
