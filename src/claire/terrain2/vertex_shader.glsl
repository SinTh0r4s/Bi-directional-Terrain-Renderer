#version 330

out float height;

uniform mat4 mvp;
uniform float offset_x;
uniform float offset_y;
uniform uint num_columns;
uniform sampler2DArray heightmap;
uniform uint lod_stride;

uvec2 id_to_grid_position(uint index) {
    uint vertices_per_row = num_columns * 2u;
    uint square_row = index / vertices_per_row;
    uint in_square_row_id = index % vertices_per_row;
    uint row_offset = in_square_row_id & 1u;
    uint row = square_row + row_offset;
    uint col = in_square_row_id >> 1u;
    if ((square_row & 1u) != 0u) {
        col = num_columns - 1u - col;
    }
    return uvec2(row, col);
}

void main() {
    uvec2 in_grid_pos = id_to_grid_position(uint(gl_VertexID));
    uint layer = 0u;
    float h = texelFetch(
        heightmap,
        ivec3(in_grid_pos.x, in_grid_pos.y, layer),
        0
    ).r;
    gl_Position = mvp * vec4(
        in_grid_pos.x * lod_stride + offset_x,
        h,
        in_grid_pos.y * lod_stride + offset_y,
        1.0
    );
    height = 1.0f;
}