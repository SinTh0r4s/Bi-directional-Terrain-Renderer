import moderngl
from pyglm import glm

from claire.camera import Camera
from claire.terrain.chunk_gpu_data import ChunkGpuData


def _render_lod_at(exponent: int, distance: float) -> bool:
    return not int(distance) >> - exponent + 18


class Chunk:
    def __init__(self, chunk_data: ChunkGpuData, chunk_size_exponent: int) -> None:
        self._chunk_data = chunk_data
        self._chunk_size_exponent = chunk_size_exponent

    def render(
        self,
        program: moderngl.Program,
        default_vao: moderngl.VertexArray,
        camera: Camera,
        textured_locations_per_exponent: dict[int, int],
    ) -> None:
        def filter_subchunk(exponent: int, subchunk_col: int, subchunk_row: int) -> None:
            aabb = self._chunk_data.lods_per_exponent[exponent].subchunks[subchunk_col][subchunk_row]
            if (
                    exponent + 1 in self._chunk_data.lods_per_exponent
                    and (aabb is None or _render_lod_at(exponent + 1, aabb.get_shortest_distance_to(camera)))
            ):
                filter_subchunk(exponent + 1, subchunk_col * 2, subchunk_row * 2)
                filter_subchunk(exponent + 1, subchunk_col * 2, subchunk_row * 2 + 1)
                filter_subchunk(exponent + 1, subchunk_col * 2 + 1, subchunk_row * 2)
                filter_subchunk(exponent + 1, subchunk_col * 2 + 1, subchunk_row * 2 + 1)
                return
            if aabb is None or not aabb.is_visible(camera):
                return
            self._render_subchunk(program, default_vao, camera, textured_locations_per_exponent, exponent, subchunk_col, subchunk_row)

        filter_subchunk(7, 0, 0)

    def _render_subchunk(
        self,
        program: moderngl.Program,
        default_vao: moderngl.VertexArray,
        camera: Camera,
        textured_locations_per_exponent: dict[int, int],
        exponent: int,
        subchunk_col: int,
        subchunk_row: int
    ) -> None:
        program["in_world_offset"].write((self._chunk_data.chunk_coords * (1 << self._chunk_size_exponent)).to_bytes())
        lod_stride = 1 << (self._chunk_size_exponent - exponent)
        program["lod_stride"] = lod_stride
        program["subchunk_offset"].write((glm.ivec2(subchunk_col, subchunk_row) * lod_stride).to_bytes())
        program["mesh_size"] = 1 << exponent
        program["chunk_id"] = self._chunk_data.chunk_sequential_id
        program["heightmap"] = textured_locations_per_exponent[exponent]
        mvp = camera.proj_matrix() * camera.view_matrix()  # model_matrix is identity
        program["mvp"].write(mvp.to_bytes())

        default_vao.render(mode=moderngl.TRIANGLES)