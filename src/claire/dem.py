from pathlib import Path

import moderngl
import numpy as np
from pyglm import glm

HEIGHTMAP = np.array([
    [0.57, 0.31, 0.44, 0.72, 0.93, 0.46, 0.00, 0.18, 0.82, 0.36],
    [0.61, 0.84, 0.10, 0.66, 0.63, 0.51, 0.84, 0.59, 0.94, 0.54],
    [0.96, 0.03, 0.12, 0.58, 0.92, 0.08, 0.18, 0.45, 0.42, 0.84],
    [0.65, 0.74, 0.12, 0.36, 0.53, 0.48, 0.42, 0.38, 0.13, 0.40],
    [0.70, 0.17, 0.79, 0.36, 0.91, 0.50, 0.06, 0.54, 0.59, 0.00],
    [0.09, 0.22, 0.82, 0.68, 0.27, 0.85, 0.90, 0.95, 0.23, 0.17],
    [0.70, 0.25, 0.07, 0.85, 0.35, 0.94, 0.04, 0.53, 0.76, 0.03],
    [0.64, 0.08, 0.89, 0.41, 0.88, 0.56, 0.29, 0.60, 0.39, 0.13],
    [0.66, 0.07, 0.14, 0.59, 0.00, 0.39, 0.68, 0.66, 0.81, 0.69],
    [0.51, 0.10, 0.09, 0.83, 0.86, 0.84, 0.95, 0.85, 0.40, 0.69],
])

_VERTEX_SHADER = Path(__file__).parent / "terrain.vert"
_FRAGMENT_SHADER = Path(__file__).parent / "terrain.frag"


class DEM:
    def __init__(self, heightmap) -> None:
        self._vertex_data, self._index_data = _get_heightmap(heightmap)


    def bind(self, ctx: moderngl.Context) -> None:
        self._program = ctx.program(
            vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
            fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8")
        )
        self._vbo = ctx.buffer(self._vertex_data.tobytes())
        self._ibo = ctx.buffer(self._index_data.tobytes())
        self._vao = ctx.vertex_array(
            self._program,
            [(self._vbo, '3f', 'in_position')],
            index_buffer=self._ibo
        )

    def _get_model_matrix(self) -> glm.mat4:
        return glm.translate(glm.vec3(-0.5, 0.0, -0.5))

    def render(self, view_proj: glm.mat4) -> None:
        mvp = view_proj * self._get_model_matrix()
        self._program['mvp'].write(mvp.to_bytes())
        self._vao.render(moderngl.TRIANGLES)

def _get_heightmap(heightmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = heightmap.shape

    # Generate Vertex Data (X, Y, Z coordinates)
    vertices = []
    for r in range(rows):
        for c in range(cols):
            x = c / (cols - 1)  # Normalize X to [0, 1]
            y = heightmap[r, c]  # Height value
            z = r / (rows - 1)  # Normalize Z to [0, 1]
            vertices.extend([x, y, z])

    vertex_data = np.array(vertices, dtype='f4')

    # Generate Index Data (Triangles for grid cells)
    indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Get indices of the 4 corners of the current quad cell
            top_left = r * cols + c
            top_right = top_left + 1
            bottom_left = (r + 1) * cols + c
            bottom_right = bottom_left + 1

            # Triangle 1
            indices.extend([top_left, bottom_left, top_right])
            # Triangle 2
            indices.extend([top_right, bottom_left, bottom_right])

    index_data = np.array(indices, dtype='u4')

    return vertex_data, index_data
