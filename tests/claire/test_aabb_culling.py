from __future__ import annotations

import pytest
from pyglm import glm

from claire.aabb import AABB
from claire.camera import HasViewProjMatrices


class FakeCamera(HasViewProjMatrices):
    def __init__(self, view: glm.mat4, proj: glm.mat4) -> None:
        self._view = view
        self._proj = proj

    def view_matrix(self) -> glm.mat4:
        return self._view

    def proj_matrix(self) -> glm.mat4:
        return self._proj


@pytest.fixture
def camera() -> HasViewProjMatrices:
    view = glm.lookAt(
        glm.vec3(0, 0, 0),
        glm.vec3(0, 0, -1),
        glm.vec3(0, 1, 0),
    )
    proj = glm.perspective(
        glm.radians(90.0),
        1.0,
        1.0,
        10.0,
    )
    return FakeCamera(view, proj)


def _get_aabb(x0: float, y0: float, z0: float) -> AABB:
    return AABB(glm.vec3(x0 - 0.5, y0 - 0.5, z0 - 0.5), glm.vec3(x0 + 0.5, y0 + 0.5, z0 + 0.5))


def test_aabb_inside_frustum(camera: HasViewProjMatrices) -> None:
    assert _get_aabb(0, 0, -3).is_visible(camera)


def test_aabb_outside_left(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(-4.5, 0, -3).is_visible(camera)


def test_aabb_outside_right(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(4.5, 0, -3).is_visible(camera)


def test_aabb_outside_top(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(0, 4.5, -3).is_visible(camera)


def test_aabb_outside_bottom(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(0, -4.5, -3).is_visible(camera)


def test_aabb_behind_camera(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(0, 0, 1).is_visible(camera)


def test_aabb_before_near_plane(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(0, 0, 0.25).is_visible(camera)


def test_aabb_beyond_far_plane(camera: HasViewProjMatrices) -> None:
    assert not _get_aabb(0, 0, 12).is_visible(camera)


def test_aabb_intersecting_near_plane_is_visible(camera: HasViewProjMatrices) -> None:
    assert _get_aabb(0, 0, -1).is_visible(camera)


def test_aabb_intersecting_far_plane_is_visible(camera: HasViewProjMatrices) -> None:
    assert _get_aabb(0, 0, -10).is_visible(camera)


def test_aabb_intersecting_left_plane_is_visible(camera: HasViewProjMatrices) -> None:
    # 90° FOV: at z=-3, left/right boundary is x=±3.
    assert _get_aabb(-3, 0, -3).is_visible(camera)


def test_aabb_intersecting_right_plane_is_visible(camera: HasViewProjMatrices) -> None:
    # 90° FOV: at z=-3, left/right boundary is x=±3.
    assert _get_aabb(3, 0, -3).is_visible(camera)


def test_aabb_intersecting_top_plane_is_visible(camera: HasViewProjMatrices) -> None:
    # 90° FOV: at z=-3, left/right boundary is x=±3.
    assert _get_aabb(0, 3, -3).is_visible(camera)


def test_aabb_intersecting_bottom_plane_is_visible(camera: HasViewProjMatrices) -> None:
    # 90° FOV: at z=-3, left/right boundary is x=±3.
    assert _get_aabb(0, -3, -3).is_visible(camera)


def test_rotated_right() -> None:
    view = glm.lookAt(
        glm.vec3(0, 0, 0),
        glm.vec3(1, 0, 0),
        glm.vec3(0, 1, 0),
    )
    proj = glm.perspective(
        glm.radians(90.0),
        1.0,
        1.0,
        10.0,
    )
    camera = FakeCamera(view, proj)
    assert _get_aabb(3, 0, 0).is_visible(camera)


def test_camera_rotated_up() -> None:
    view = glm.lookAt(
        glm.vec3(0, 0, 0),
        glm.vec3(0, 1, -0.01),  # cannot look exactly at camera up
        glm.vec3(0, 1, 0),
    )
    proj = glm.perspective(
        glm.radians(90.0),
        1.0,
        1.0,
        10.0,
    )
    camera = FakeCamera(view, proj)
    assert _get_aabb(0, 3, 0).is_visible(camera)


def test_camera_flipped() -> None:
    view = glm.lookAt(
        glm.vec3(0, 0, 0),
        glm.vec3(0, 0, 1),
        glm.vec3(0, 1, 0),
    )
    proj = glm.perspective(
        glm.radians(90.0),
        1.0,
        1.0,
        10.0,
    )
    camera = FakeCamera(view, proj)
    assert _get_aabb(0, 0, 3).is_visible(camera)
