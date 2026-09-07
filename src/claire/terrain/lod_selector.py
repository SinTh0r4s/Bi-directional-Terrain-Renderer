import math
from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, TypeVar, Generic

import numpy as np
from pyglm import glm

from claire.aabb import AABB
from claire.camera import HasCameraPositionResolutionFovNearplane
from claire.terrain.numpy_types import HeightmapData


@dataclass
class LodData:
    aabb: AABB
    max_error_y: float


_T = TypeVar("_T")


class LodSelector(ABC, Generic[_T]):
    def should_refine(self, lod_data: _T) -> Literal["refine", "render", "use_previous"]:...


class MaxErrorLodSelector(LodSelector[LodData]):
    def __init__(self, camera: HasCameraPositionResolutionFovNearplane, max_error_in_px: float, hysteresis_factor: float = 0.1) -> None:
        self.camera = camera
        self._max_error_in_px = max_error_in_px
        self._hysteresis_factor = hysteresis_factor
        self._render_threshold, self._refine_threshold = self._get_thresholds(camera)

    def _get_thresholds(self, camera: HasCameraPositionResolutionFovNearplane) -> tuple[float, float]:
        factor_k = camera.resolution.y / (2 * math.tan(math.radians(camera.fov_deg) / 2))
        render_threshold = self._max_error_in_px * (1 - self._hysteresis_factor) / factor_k
        refine_threshold = self._max_error_in_px / factor_k
        return render_threshold, refine_threshold

    def update_camera(self, camera: HasCameraPositionResolutionFovNearplane) -> None:
        self.camera = camera
        self._render_threshold, refine_threshold = self._get_thresholds(camera)

    def should_refine(self, lod_data: LodData) -> Literal["refine", "render", "use_previous"]:
        distance = lod_data.aabb.get_shortest_distance_to(self.camera)
        if distance < self.camera.near_plane:
            return "render"
        ratio = lod_data.max_error_y / distance
        if ratio < self._render_threshold:
            return "render"
        if ratio >= self._refine_threshold:
            return "refine"
        return "use_previous"


def _inside_triangle(barycentric_coords: glm.vec3) -> bool:
    return (barycentric_coords.x >= 0.0) and (barycentric_coords.y >= 0.0) and (barycentric_coords.z >= 0.0)


def _get_barycentric_coords(p: glm.vec2, a: glm.vec2, b: glm.vec2, c: glm.vec2) -> glm.vec3:
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = glm.dot(v0, v0)
    d01 = glm.dot(v0, v1)
    d11 = glm.dot(v1, v1)
    d20 = glm.dot(v2, v0)
    d21 = glm.dot(v2, v1)

    inv_denom = 1 / (d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom
    return glm.vec3(1.0 - v - w, v, w)


def _interpolate_height(a: float, b: float, c: float, d: float, col: float, row: float) -> float:
    sample = glm.vec2(col, row)
    point_a = glm.vec2(0, 0)
    point_b = glm.vec2(0, 1)
    point_c = glm.vec2(1, 0)
    point_d = glm.vec2(1, 1)

    barycentric_coords = _get_barycentric_coords(sample, point_a, point_b, point_c)
    if _inside_triangle(barycentric_coords):
        return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z
    barycentric_coords = _get_barycentric_coords(sample, point_b, point_c, point_d)
    return b * barycentric_coords.x + c * barycentric_coords.y + d * barycentric_coords.z


def extract_lod_introduced_max_error(full_heightmap: HeightmapData, offset: glm.ivec2, lod_stride: int, lod_width: int) -> float:
    full_height_size = lod_stride * lod_width
    full_cols, full_rows = full_heightmap.shape
    usable_cols = min(full_cols, offset.x + full_height_size + 1) - offset.x
    usable_rows = min(full_rows, offset.y + full_height_size + 1) - offset.y
    lod_cols = math.ceil(usable_cols / lod_stride)
    lod_rows = math.ceil(usable_rows / lod_stride)
    lod_heightmap = full_heightmap[
        offset.x: offset.x + lod_cols * lod_stride: lod_stride,
        offset.y: offset.y + lod_rows * lod_stride: lod_stride,
    ]

    max_error = 0.0
    # We don't need to calculate the error for the overlap of the next tile
    for col in range(min(lod_cols * lod_stride - 1, full_height_size)):
        for row in range(min(lod_rows * lod_stride - 1, full_height_size)):
            true_height = full_heightmap[col + offset.x, row + offset.y]
            col_fract, col_int = math.modf(col / lod_stride)
            row_fract, row_int = math.modf(row / lod_stride)
            col_int = int(col_int)
            row_int = int(row_int)

            a = lod_heightmap[col_int, row_int]
            if col_fract == 0:
                if row_fract == 0:
                    continue  # no error as this is where the full map was sampled
                b = lod_heightmap[col_int, row_int + 1]
                if a <= 0 or b <= 0:
                    continue
                max_error = max(max_error, true_height - a * (1 - row_fract) - b * row_fract)
            elif row_fract == 0:
                c = lod_heightmap[col_int + 1, row_int]
                if a <= 0 or c <= 0:
                    continue
                max_error = max(max_error, true_height - a * (1 - col_fract) - c * col_fract)
            else:
                b = lod_heightmap[col_int, row_int + 1]
                c = lod_heightmap[col_int + 1, row_int]
                d = lod_heightmap[col_int + 1, row_int + 1]
                if a <= 0 or b <= 0 or c <= 0 or d <= 0:
                    continue

                error_1 = abs(true_height - _interpolate_height(a, b, c, d, col_fract, row_fract))
                error_2 = abs(true_height - _interpolate_height(c, d, a, b, 1 - col_fract, row_fract))

                max_error = max(max_error, error_1, error_2)
    return max_error


def extract_lod_aabb(full_heightmap: HeightmapData, offset: glm.ivec2, lod_stride: int, lod_width: int) -> Optional[AABB]:
    full_height_size = lod_stride * lod_width
    full_cols, full_rows = full_heightmap.shape
    max_col = min(full_cols, offset.x + full_height_size + 1) - offset.x
    max_row = min(full_rows, offset.y + full_height_size + 1) - offset.y
    lod_heightmap = full_heightmap[offset.x: max_col, offset.y: max_row]
    max_height = np.max(lod_heightmap)
    if max_height <= 0:
        return None
    min_height = np.min(lod_heightmap[lod_heightmap > 0])
    return AABB(glm.vec3(offset.x, min_height, offset.y), glm.vec3(max_col, max_height, max_row))


def create_load_data(heightmap: HeightmapData, terrain_offset: glm.ivec2, lod_stride: int, lod_size: int) -> Optional[LodData]:
    aabb = extract_lod_aabb(heightmap, terrain_offset, lod_stride, lod_size)
    if aabb is None:
        return None
    max_error_y = extract_lod_introduced_max_error(heightmap, terrain_offset, lod_stride, lod_size)
    return LodData(aabb, max_error_y)
