#version 330

// double curly brackets to escape for string baking!
// baked constants:
// LOD_COUNT
// MESH_SIZE_EXPONENT
// TEXTURE_TILE_SIZE_EXPONENT

out float height;
out vec3 normal;
out vec3 position;

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

bool get_height(ivec2 heightmap_pos, out float height) {{
    ivec2 tile_coord = heightmap_pos >> {TEXTURE_TILE_SIZE_EXPONENT};
    int tile_id = texelFetch(tile_id_lookup, tile_coord, 0).r;
    bool valid_tile = tile_id != -1;
    tile_id = int(valid_tile) * tile_id;  // 0 is fallback to keep pipeline alive
    ivec2 in_tile_pos = heightmap_pos - (tile_coord << {TEXTURE_TILE_SIZE_EXPONENT});
    ivec3 sample_uv = ivec3(in_tile_pos >> lod_level, tile_id);
    height = texelFetch(heightmap[lod_level], sample_uv, 0).r;
    bool valid_height = height > 0;
    return valid_tile && valid_height;
}}

bool get_height_with_normal(ivec2 terrain_pos, out float height, out vec3 normal) {{
    float height_center = 0.0;
    bool valid_center = get_height(terrain_pos, height_center);

    int weight_ns = 2;
    float height_north = 0.0;
    bool valid_north = get_height(terrain_pos + ivec2(0, -1), height_north);
    height_north = int(valid_north) * height_north + (1 - int(valid_north)) * height_center;
    weight_ns = weight_ns - int(!valid_north);
    float height_south = 0.0;
    bool valid_south = get_height(terrain_pos + ivec2(0, 1), height_south);
    height_south = int(valid_south) * height_south + (1 - int(valid_south)) * height_center;
    weight_ns = weight_ns - int(!valid_south);
    bool valid_ns = weight_ns != 0;

    int weight_we = 2;
    float height_west = 0.0;
    bool valid_west = get_height(terrain_pos + ivec2(-1, 0), height_west);
    height_west = int(valid_west) * height_west + (1 - int(valid_west)) * height_center;
    weight_we = weight_we - int(!valid_west);
    float height_east = 0.0;
    bool valid_east = get_height(terrain_pos + ivec2(1, 0), height_east);
    height_east = int(valid_east) * height_east + (1 - int(valid_east)) * height_center;
    weight_we = weight_we - int(!valid_east);
    bool valid_we = weight_we != 0;

    float derivative_ns = (height_south - height_north) / weight_ns;
    float derivative_we = (height_east - height_west) / weight_we;
    // cross product of both derivative vectors along their axes
    normal = normalize(vec3(-derivative_we, 1.0, -derivative_ns));
    height = height_center;
    return valid_center && valid_ns && valid_we;
}}

void main() {{
    ivec2 terrain_pos = get_terrain_pos(gl_VertexID);
    bool valid = get_height_with_normal(terrain_pos, height, normal);
    // NaN effectively marks all triangles using this vertex to be culled
    height = valid ? height : 0.0 / 0.0;
    position = vec3(terrain_pos.x, height, terrain_pos.y);
    gl_Position = mvp * vec4(position, 1.0);
    height = 1.0f;
}}