#version 330

in float height;

out vec4 f_color;

void main() {
    f_color = vec4(height * 0.4, height * 0.8 + 0.2, 0.3, 1.0);
}