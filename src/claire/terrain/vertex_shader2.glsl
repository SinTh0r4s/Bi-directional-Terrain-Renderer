#version 330

// double curly brackets to escape for string baking!
// baked constants:
// LOD_COUNT
// MESH_SIZE_EXPONENT
// TEXTURE_TILE_SIZE_EXPONENT

out float height;

uniform sampler2DArray heightmap[{LOD_COUNT}];
uniform isampler2D tile_id_lookup;
uniform mat4 mvp;
uniform int lod_level;
uniform ivec2 offset;
uniform ivec4 neighbor_lod_nwse;

const int mesh_size = (1 << {MESH_SIZE_EXPONENT}) + 1;

ivec2 get_terrain_pos(int vertex_id) {{
    ivec2 mesh_pos = ivec2(vertex_id / mesh_size, vertex_id % mesh_size);
    return offset + (mesh_pos << lod_level);
}}

float get_height(ivec2 heightmap_pos) {{
    ivec2 tile_coord = heightmap_pos >> {TEXTURE_TILE_SIZE_EXPONENT};
    int tile_id = texelFetch(tile_id_lookup, tile_coord, 0).r;
    if (tile_id == -1) {{
        return -1.0;
    }}
    ivec2 in_tile_pos = heightmap_pos - (tile_coord << {TEXTURE_TILE_SIZE_EXPONENT});
    ivec3 sample_uv = ivec3(in_tile_pos >> lod_level, tile_id);
    return texelFetch(heightmap[lod_level], sample_uv, 0).r;
}}

void main() {{
    ivec2 terrain_pos = get_terrain_pos(gl_VertexID);
    height = get_height(terrain_pos);
    if (height <= 0.0f) {{
        height = 0.0f / 0.0f;  // NaN effectively marks all triangles using this vertex to be culled
    }}
    gl_Position = mvp * vec4(terrain_pos.x, height, terrain_pos.y, 1.0);
    height = 1.0f;
}}