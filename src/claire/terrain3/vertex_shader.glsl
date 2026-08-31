#version 330

out float height;

uniform ivec2 in_world_offset;
uniform int lod_stride;
uniform ivec2 subchunk_offset;
uniform int mesh_size;
uniform int chunk_id;
uniform sampler2DArray heightmap;
uniform mat4 mvp;

void main() {
    ivec2 mesh_pos = ivec2(gl_VertexID / mesh_size, gl_VertexID % mesh_size);
    ivec2 subchunk_pos = subchunk_offset + mesh_pos;
    height = texelFetch(
        heightmap,
        ivec3(subchunk_pos.x, subchunk_pos.y, chunk_id),
        0
    ).r;
    ivec2 terrain_pos = in_world_offset + subchunk_pos * lod_stride;
    gl_Position = mvp * vec4(terrain_pos.x, height, terrain_pos.y, 1.0);
    height = 1.0f;
}