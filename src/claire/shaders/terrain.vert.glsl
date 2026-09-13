#version 330

#bake LOD_COUNT
#bake MESH_SIZE_EXPONENT
#bake TEXTURE_TILE_SIZE_EXPONENT


out float height;
out vec3 normal;
out vec3 position;

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

bool get_height(ivec2 heightmap_pos, out float height) {
    ivec2 tile_coord = heightmap_pos >> TEXTURE_TILE_SIZE_EXPONENT;
    int tile_id = texelFetch(tile_id_lookup, tile_coord, 0).r;
    bool valid_tile = tile_id != -1;
    tile_id = int(valid_tile) * tile_id;  // 0 is fallback to keep pipeline alive
    ivec2 in_tile_pos = heightmap_pos - (tile_coord << TEXTURE_TILE_SIZE_EXPONENT);
    ivec3 sample_uv = ivec3(in_tile_pos >> lod_level, tile_id);
    height = texelFetch(heightmap[lod_level], sample_uv, 0).r;
    bool valid_height = height > 0;
    return valid_tile && valid_height;
}

bool get_connected_height(ivec2 heightmap_pos, ivec2 mesh_pos, out float height) {
    ivec2 sample_lod = heightmap_pos >> lod_level;

    bool activate_north = mesh_pos.y == 0;
    bool activate_south = mesh_pos.y == mesh_size - 1;
    int lod_delta_x = int(activate_north) * neighbor_lod_nwse.x + int(activate_south) * neighbor_lod_nwse.z + int(!(activate_north || activate_south)) * lod_level - lod_level;
    int lod_delta_stride_x = 1 << lod_delta_x;
    int sample_lod_x1 = sample_lod.x & ~(lod_delta_stride_x - 1);
    int sample_lod_x2 = (sample_lod.x + lod_delta_stride_x - 1) & ~(lod_delta_stride_x - 1);  // identical to x1 if sample_uv.x == x1; otherwise + stride
    float mix_x = float(sample_lod.x - sample_lod_x1) / float(lod_delta_stride_x);

    bool activate_west = mesh_pos.x == 0;
    bool activate_east = mesh_pos.x == mesh_size - 1;
    int lod_delta_y = int(activate_west) * neighbor_lod_nwse.y + int(activate_east) * neighbor_lod_nwse.w + int(!(activate_west || activate_east)) * lod_level - lod_level;
    int lod_delta_stride_y = 1 << lod_delta_y;
    int sample_lod_y1 = sample_lod.y & ~(lod_delta_stride_y - 1);
    int sample_lod_y2 = (sample_lod.y + lod_delta_stride_y - 1) & ~(lod_delta_stride_y - 1);  // identical to y1 if sample_uv.y == y1; otherwise + stride
    float mix_y = float(sample_lod.y - sample_lod_y1) / float(lod_delta_stride_y);

    ivec2 sample_world_1 = ivec2(sample_lod_x1, sample_lod_y1) << lod_level;
    ivec2 sample_world_2 = ivec2(sample_lod_x2, sample_lod_y2) << lod_level;

    float height_1;
    bool valid_1 = get_height(sample_world_1, height_1);
    float height_2;
    bool valid_2 = get_height(sample_world_2, height_2);

    float mix = max(mix_x, mix_y);  // only >0 if interpolation is actually happening for the axis & axis are interpolating exclusively
    height = (1 - mix) * height_1 + mix * height_2;
    return valid_1 && valid_2;
}

bool get_height_with_normal(ivec2 terrain_pos, ivec2 mesh_pos, out float height, out vec3 normal) {
    float height_center;
    bool valid_center = get_connected_height(terrain_pos, mesh_pos, height_center);

    int weight_ns = 2;
    float height_north;
    bool valid_north = get_connected_height(terrain_pos + ivec2(0, -1), mesh_pos, height_north);
    height_north = int(valid_north) * height_north + (1 - int(valid_north)) * height_center;
    weight_ns = weight_ns - int(!valid_north);
    float height_south;
    bool valid_south = get_connected_height(terrain_pos + ivec2(0, 1), mesh_pos, height_south);
    height_south = int(valid_south) * height_south + (1 - int(valid_south)) * height_center;
    weight_ns = weight_ns - int(!valid_south);
    bool valid_ns = weight_ns != 0;

    int weight_we = 2;
    float height_west;
    bool valid_west = get_connected_height(terrain_pos + ivec2(-1, 0), mesh_pos, height_west);
    height_west = int(valid_west) * height_west + (1 - int(valid_west)) * height_center;
    weight_we = weight_we - int(!valid_west);
    float height_east;
    bool valid_east = get_connected_height(terrain_pos + ivec2(1, 0), mesh_pos, height_east);
    height_east = int(valid_east) * height_east + (1 - int(valid_east)) * height_center;
    weight_we = weight_we - int(!valid_east);
    bool valid_we = weight_we != 0;

    float derivative_ns = (height_south - height_north) / (weight_ns << lod_level);
    float derivative_we = (height_east - height_west) / (weight_we << lod_level);
    // cross product of both derivative vectors along their axes
    normal = normalize(vec3(-derivative_we, 1.0, -derivative_ns));
    height = height_center;
    return valid_center && valid_ns && valid_we;
}

void main() {
    ivec2 mesh_pos;
    ivec2 terrain_pos = get_terrain_pos(gl_VertexID, mesh_pos);
    bool valid = get_height_with_normal(terrain_pos, mesh_pos, height, normal);

    position = vec3(terrain_pos.x, height, terrain_pos.y);
    gl_Position = mvp * vec4(position, 1.0);
    // NaN effectively marks all triangles using this vertex to be culled - might not work on all hardware!
    gl_Position = valid ? gl_Position : vec4(0.0 / 0.0, 0.0 / 0.0, 0.0 / 0.0, -1);

    height = 1.0f;
}