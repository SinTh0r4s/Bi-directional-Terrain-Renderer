#version 330
in uvec2 in_grid_pos;
in float in_height;

out float height;

uniform mat4 mvp;
uniform float latitude_left;
uniform float longitude_bottom;
uniform uint stride;

void main() {
    gl_Position = mvp * vec4(in_grid_pos.y * stride + latitude_left, in_height, in_grid_pos.x * stride + longitude_bottom, 1.0);
    height = 1.0f;
}