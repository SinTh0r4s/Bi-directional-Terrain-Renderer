import pytest
from pyglm import glm

from claire.aabb import AABB
from claire.camera import HasCameraPosition


class _FakeCamera(HasCameraPosition):
    def __init__(self, position: glm.vec3) -> None:
        self.position = position

_AABB = AABB(position_min=glm.vec3(0, 0, 0), position_max=glm.vec3(10, 10, 10))


def test_camera_in_box() -> None:
    camera = _FakeCamera(glm.vec3(5, 5, 5))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(0)

def test_camera_above_box() -> None:
    camera = _FakeCamera(glm.vec3(5, 5, 15))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(5)


def test_camera_below_box() -> None:
    camera = _FakeCamera(glm.vec3(5, 5, -6))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(6)


def test_camera_north_box() -> None:
    camera = _FakeCamera(glm.vec3(5, -4, 5))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(4)


def test_camera_south_box() -> None:
    camera = _FakeCamera(glm.vec3(5, 15.5, 5))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(5.5)


def test_camera_west_box() -> None:
    camera = _FakeCamera(glm.vec3(-7, 5, 5))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(7)


def test_camera_east_box() -> None:
    camera = _FakeCamera(glm.vec3(18, 5, 5))
    assert _AABB.get_shortest_distance_to(camera) == pytest.approx(8)
