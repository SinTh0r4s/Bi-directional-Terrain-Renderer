from __future__ import annotations

import pytest
from pyglm import glm

from claire.aabb import AABB
from claire.camera import HasCameraPositionResolutionFovNearplane
from claire.terrain.lod_selector import LodData, MaxErrorLodSelector


class _Camera(HasCameraPositionResolutionFovNearplane):
    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 10.0),
        resolution: tuple[int, int] = (1920, 1080),
        fov_deg: float = 60.0,
        near_plane: float = 0.1,
    ):
        self.position = glm.vec3(*position)
        self.resolution = glm.ivec2(*resolution)
        self.fov_deg = fov_deg
        self.near_plane = near_plane


class _TestLodData(LodData):
    def __init__(self, aabb: AABB, max_error_y: float):
        self.aabb = aabb
        self.max_error_y = max_error_y


@pytest.fixture
def aabb() -> AABB:
    return AABB(
        glm.vec3(-1.0, -1.0, -1.0),
        glm.vec3(1.0, 1.0, 1.0),
    )


@pytest.fixture
def selector() -> MaxErrorLodSelector:
    return MaxErrorLodSelector(
        _Camera(),
        max_error_in_px=10.0,
        hysteresis_factor=0.1,
    )


def test_error_below_threshold_renders(selector: MaxErrorLodSelector, aabb: AABB) -> None:
    assert selector.should_refine(_TestLodData(aabb, 0.01)) == "render"


def test_error_above_threshold_refines(selector: MaxErrorLodSelector, aabb: AABB) -> None:
    assert selector.should_refine(_TestLodData(aabb, 100.0)) == "refine"


def test_error_at_threshold_has_defined_behavior(selector: MaxErrorLodSelector, aabb: AABB) -> None:
    result = selector.should_refine(_TestLodData(aabb, 10.0))

    assert result in {"render", "refine", "use_previous"}


def test_error_inside_hysteresis_returns_use_previous(selector: MaxErrorLodSelector, aabb: AABB) -> None:
    # First establish that this tile is well above the threshold.
    assert selector.should_refine(_TestLodData(aabb, 0.1)) == "refine"
    assert selector.should_refine(_TestLodData(aabb, 0.08)) == "render"

    # Now move into the hysteresis region.
    result = selector.should_refine(_TestLodData(aabb, 0.09))

    assert result == "use_previous"


def test_div_by_zero_protection_too_close_to_camera(aabb: AABB) -> None:
    camera = _Camera(position=(0.0, 0.0, 1.0), near_plane=1)
    selector = MaxErrorLodSelector(camera, 10.0, 0.0)
    assert selector.should_refine(_TestLodData(aabb, 99999999999999)) == "render"


def test_update_camera_changes_selection(aabb: AABB) -> None:
    camera = _Camera(position=(0.0, 0.0, 100.0))
    selector = MaxErrorLodSelector(camera, 10.0, 0.0)

    error = 0.5
    far_result = selector.should_refine(_TestLodData(aabb, error))
    selector.update_camera(_Camera(position=(0.0, 0.0, 3.0)))
    near_result = selector.should_refine(_TestLodData(aabb, error))
    assert far_result != "refine"
    assert near_result == "refine"
