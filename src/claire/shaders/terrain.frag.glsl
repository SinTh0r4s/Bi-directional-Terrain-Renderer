#version 330

in vec3 normal;
in vec3 position;

out vec4 f_color;

uniform vec3 camera_position;
uniform vec3 terrain_default_color;
uniform vec3 sun_direction;
uniform vec3 sun_color;

const float AMBIENT_WEIGHT = 0.20;
const float DIFFUSE_WEIGHT = 0.85;
const float SPECULAR_WEIGHT = 0.05;
const float SHININESS = 16.0;

void main() {
    vec3 n = normalize(normal);
    vec3 view = normalize(camera_position - position);

    float diffuse = max(dot(n, sun_direction), 0.0);
    vec3 reflected_light = reflect(-sun_direction, n);
    float specular = pow(max(dot(view, reflected_light), 0.0), SHININESS);

    vec3 lighting =
        terrain_default_color * AMBIENT_WEIGHT +
        terrain_default_color * sun_color * diffuse * DIFFUSE_WEIGHT +
        sun_color * specular * SPECULAR_WEIGHT;

    f_color = vec4(lighting, 1.0);
}