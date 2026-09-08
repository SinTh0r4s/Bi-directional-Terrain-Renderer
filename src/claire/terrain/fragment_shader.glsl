#version 330

in float height;
in vec3 normal;
in vec3 position;

out vec4 f_color;

uniform vec3 camera_position;

const vec3 SUN_DIR = normalize(vec3(-0.6, 0.8, -0.35));
const vec3 SUN_COLOR = vec3(1.0, 0.93, 0.82);

const vec3 TERRAIN_COLOR = vec3(0.28, 0.42, 0.20);

const float AMBIENT_WEIGHT = 0.15;
const float DIFFUSE_WEIGHT = 0.75;
const float SPECULAR_WEIGHT = 0.25;
const float SHININESS = 32.0;

void main() {
    vec3 n = normalize(normal);
    vec3 view = normalize(camera_position - position);

    float diffuse = max(dot(n, SUN_DIR), 0.0);
    vec3 reflected_light = reflect(-SUN_DIR, n);
    float specular = pow(max(dot(view, reflected_light), 0.0), SHININESS);

    vec3 lighting =
        TERRAIN_COLOR * AMBIENT_WEIGHT +
        TERRAIN_COLOR * SUN_COLOR * diffuse * DIFFUSE_WEIGHT +
        SUN_COLOR * specular * SPECULAR_WEIGHT;

    f_color = vec4(lighting, 1.0);
}