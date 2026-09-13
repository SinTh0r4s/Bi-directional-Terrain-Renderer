#version 330

#include "includes/heightmap_constants.glsl"
#include "includes/sample_heightmap.glsl"

uniform sampler2DArray heightmap[LOD_COUNT];
uniform isampler2D tile_id_lookup;
uniform mat4 mvp;
uniform int lod_level;
uniform ivec2 offset;
uniform ivec4 neighbor_lod_nwse;

const int mesh_size = (1 << MESH_SIZE_EXPONENT) + 1;

ivec2 get_terrain_pos(int vertex_id, out ivec2 mesh_pos) {
    mesh_pos = ivec2(vertex_id / mesh_size, vertex_id % mesh_size);
    return offset + (mesh_pos << lod_level);
}

void main() {
    ivec2 mesh_pos;
    ivec2 terrain_pos = get_terrain_pos(gl_VertexID, mesh_pos);
    float height;
    bool valid = get_connected_height(heightmap, tile_id_lookup, terrain_pos, lod_level, mesh_pos, neighbor_lod_nwse, height);
    gl_Position = mvp * vec4(terrain_pos.x, height, terrain_pos.y, 1.0);
    // NaN effectively marks all triangles using this vertex to be culled - might not work on all hardware!
    gl_Position = valid ? gl_Position : vec4(0.0 / 0.0, 0.0 / 0.0, 0.0 / 0.0, -1);
}
