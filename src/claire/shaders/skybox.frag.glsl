#version 330

uniform mat4 inv_proj;
uniform mat4 inv_view;
uniform ivec2 viewport_width_height;

out vec4 f_color;

const vec3 SKY_ZENITH = vec3(0.075, 0.15, 0.30);
const vec3 SKY_MIDDLE = vec3(0.22, 0.39, 0.60);
const vec3 SKY_HORIZON = vec3(0.78, 0.74, 0.65);

const float SUN_RADIUS = 1.5;
const float SUN_HALO_POWER = 64.0;
const float SUN_HALO_STRENGTH = 0.25;

const vec3 HORIZON_COLOR = vec3(0.95, 0.72, 0.48);
const float HORIZON_STRENGTH = 0.10;
const float HORIZON_WIDTH = 10.0;

// TODO: sync with terrain
const vec3 SUN_DIR = normalize(vec3(-0.6, 0.8, -0.35));
const vec3 SUN_COLOR = vec3(1.0, 0.93, 0.82);

vec3 get_ray_direction() {
    vec2 ndc = vec2(
        (gl_FragCoord.x / viewport_width_height.x) * 2.0 - 1.0,
        (gl_FragCoord.y / viewport_width_height.y) * 2.0 - 1.0
    );

    vec4 ray_view = inv_proj * vec4(ndc, 1.0, 1.0);
    ray_view /= ray_view.w;

    return normalize((inv_view * ray_view).xyz);
}

void main() {
    vec3 ray = normalize(get_ray_direction());
    float y = ray.y;

    // Sky gradient
    vec3 sky;
    if (y >= 0.0) {
        float t = smoothstep(0.0, 1.0, y);
        sky = mix(SKY_MIDDLE, SKY_ZENITH, t);
    }
    else {
        float t = smoothstep(-0.35, 0.0, y);
        sky = mix(HORIZON_COLOR, SKY_MIDDLE, t);
    }

    // warm atmosphere
    float horizon = exp(-abs(y) * HORIZON_WIDTH);

    sky += HORIZON_COLOR * horizon * HORIZON_STRENGTH;

    // sun disk
    float sun_dot = dot(ray, normalize(SUN_DIR));
    float sun_radius = radians(SUN_RADIUS);
    float sun_disk = smoothstep(
        cos(sun_radius),
        cos(sun_radius * 0.65),
        sun_dot
    );

    sky += SUN_COLOR * sun_disk;

    // sun halo
    float halo = pow(max(sun_dot, 0.0), SUN_HALO_POWER);

    sky+= SUN_COLOR * halo * SUN_HALO_STRENGTH;

    f_color = vec4(sky, 1.0);
}
