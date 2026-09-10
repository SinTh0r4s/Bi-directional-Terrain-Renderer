#version 330

in vec3 in_position;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}
