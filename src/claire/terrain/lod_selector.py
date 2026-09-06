import math
from abc import ABC
from typing import Literal

from claire.aabb import AABB
from claire.camera import HasCameraPositionResolutionFovNearplane

class LodData(ABC):
    aabb: AABB
    max_error_y: float


class LodSelector(ABC):
    def should_refine(self, lod_data: LodData) -> Literal["refine", "render", "use_previous"]:...


class MaxErrorLodSelector:
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