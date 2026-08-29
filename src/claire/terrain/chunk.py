from dataclasses import dataclass

import moderngl
import numpy as np
from pyglm import glm

from claire.camera import Camera
from claire.terrain.load_data import DataChunk


def _create_vertex_data(heightmap: np.ndarray) -> np.ndarray:
    rows, cols = heightmap.shape
    num_vertices = rows * cols
    vertex_dtype = np.dtype([
        ('x', 'u2'),
        ('z', 'u2'),
        ('y', 'f4')
    ])
    vbo_array = np.empty(num_vertices, dtype=vertex_dtype)
    r_indices, c_indices = np.mgrid[0:rows, 0:cols]
    vbo_array['x'] = c_indices.flatten()
    vbo_array['z'] = r_indices.flatten()
    vbo_array['y'] = heightmap.flatten()

    return vbo_array


def _create_index_buffer(rows: float, cols: float) -> np.ndarray:
    num_quads = (rows - 1) * (cols - 1)
    num_indices = num_quads * 6  # 2 triangles * 3 indices per quad

    index_data = np.empty(num_indices, dtype='u4')

    # Generate vertex corner indices for every quad cell
    r_indices, c_indices = np.mgrid[0:rows - 1, 0:cols - 1]
    top_left = (r_indices * cols + c_indices).flatten()
    top_right = top_left + 1
    bottom_left = top_left + cols
    bottom_right = bottom_left + 1

    index_data[0::6] = top_left
    index_data[1::6] = bottom_left
    index_data[2::6] = top_right
    index_data[3::6] = top_right
    index_data[4::6] = bottom_left
    index_data[5::6] = bottom_right

    return index_data


@dataclass
class _LodData:
    vertex_data: np.ndarray
    index_data: np.ndarray

    @staticmethod
    def from_heightmap(heightmap: np.ndarray) -> "_LodData":
        return _LodData(
            vertex_data=_create_vertex_data(heightmap),
            index_data=_create_index_buffer(heightmap.shape[0], heightmap.shape[1]),
        )

@dataclass
class _LOD:
    vertex_buffer: moderngl.Buffer
    index_buffer: moderngl.Buffer
    vao: moderngl.VertexArray


def _select_lod(lods: list[_LOD], distance: float) -> tuple[_LOD, int]:
    if distance < 1000:
        return lods[0], 1
    if distance < 2000:
        return lods[1], 4
    return lods[2], 8


# TODO: actively delete arrays after they are not required any more
class Chunk:
    def __init__(self, data_chunk: DataChunk) -> None:
        self._latitude_left = data_chunk.latitude_left
        self._longitude_bottom = data_chunk.longitude_bottom
        self._width, self._height, *_ = data_chunk.heightmap.shape

        self._lod_data = [
            _LodData.from_heightmap(data_chunk.heightmap),
            _LodData.from_heightmap(data_chunk.heightmap[::4, ::4]),
            _LodData.from_heightmap(data_chunk.heightmap[::8, ::8]),
        ]
        self._lods: list[_LOD] = []

    def upload(self, ctx: moderngl.Context, terrain_program: moderngl.Program) -> None:
        def upload_data(lod_data: _LodData) -> _LOD:
            vbo = ctx.buffer(lod_data.vertex_data.tobytes())
            ibo = ctx.buffer(lod_data.index_data.tobytes())
            vao = ctx.vertex_array(
                terrain_program,
                [(vbo, '2u2 1f4', 'in_grid_pos', 'in_height')],
                index_buffer=ibo
            )
            return _LOD(vertex_buffer=vbo, index_buffer=ibo, vao=vao)

        self._lods = [upload_data(lod_data) for lod_data in self._lod_data]

    def _model_matrix(self) -> glm.mat4:
        return glm.identity(glm.mat4)

    def render(self, camera: Camera, terrain_program: moderngl.Program) -> None:
        aabb_min = glm.vec3(self._latitude_left, 0, self._longitude_bottom)
        aabb_max = aabb_min + glm.vec3(self._width, 3_000, self._height)
        if not camera.is_visible(aabb_min, aabb_max):
            return
        distance = camera.min_distance_to_camera(aabb_min, aabb_max)
        lod, stride = _select_lod(self._lods, distance)

        mvp = camera.proj_matrix() * camera.view_matrix() * self._model_matrix()
        terrain_program['mvp'].write(mvp.to_bytes())
        terrain_program['latitude_left'] = self._latitude_left
        terrain_program['longitude_bottom'] = self._longitude_bottom
        terrain_program['stride'] = stride
        lod.vao.render(moderngl.TRIANGLES)