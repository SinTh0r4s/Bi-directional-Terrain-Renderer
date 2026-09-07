import numpy as np
import pytest
from pyglm import glm

from claire.terrain.lod_selector import extract_lod_introduced_max_error


def test_identical_heightmaps_have_zero_error():
    full = np.array([
        [1, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8]
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 1, 4) == pytest.approx(0.0)


def test_planar_surface_has_zero_error():
    """
    A planar heightmap is reproduced exactly by either triangle
    interpolation, regardless of diagonal direction.
    """
    y, x = np.mgrid[0:5, 0:5]
    full = 2.0 * x + 3.0 * y
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 2) == pytest.approx(0.0)


def test_diagonally_hidden_values() -> None:
    full = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1],
        [1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 2) == pytest.approx(1)


def test_horizontally_hidden_values() -> None:
    full = np.array([
        [1, 2, 1, 2, 1],
        [1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 2) == pytest.approx(1)


def test_vertically_hidden_values() -> None:
    full = np.array([
        [1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1],
        [1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 2) == pytest.approx(1)


def test_both_triangle_diagonals_are_considered():
    """
    The two possible quad triangulations must both be considered.

        A ----- B
        | \\    |
        |   \\  |
        |     \\|
        C ----- D

    versus

        A ----- B
        |     / |
        |   /   |
        | /     |
        C ----- D

    Construct a case where the two triangulations produce different
    errors. The result must be the worse of the two.
    """
    full = np.array([
        [1, 1, 10, 1, 1],
        [1, 10, 1, 1, 1],
        [10, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 2) == pytest.approx(9)


def test_constant_surface_has_zero_error():
    full = np.full((5, 5), 42.5)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 2) == pytest.approx(0.0)


def test_larger_lod() -> None:
    full = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 4, 2) == pytest.approx(1)


def test_offset_lod() -> None:
    full = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(4, 4), 2, 2) == pytest.approx(0)


def test_missing_overlap() -> None:
    """ Last row and column cannot be considered """
    full = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 3],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 4) == pytest.approx(1)


def test_even_size_partial_tile() -> None:
    """ Last row and column cannot be considered if even """
    full = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1, 1],
        [1, 1, 1, 1, 1, 3],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 4) == pytest.approx(1)


def test_odd_size_partial_tile() -> None:
    """ Last row and column ARE considered if odd """
    full = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 2, 1, 1],
        [1, 1, 1, 2, 1, 1, 1],
        [1, 1, 1, 1, 1, 2, 3],
        [1, 1, 1, 1, 1, 3, 1],
    ], dtype=np.float32)
    assert extract_lod_introduced_max_error(full, glm.ivec2(0, 0), 2, 4) == pytest.approx(2)
