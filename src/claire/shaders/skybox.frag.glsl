#version 330

uniform mat4 inv_proj;
uniform mat4 inv_view;
uniform ivec2 viewport_width_height;
uniform vec3 sky_zenith_color;
uniform vec3 sky_main_color;
uniform vec3 sky_horizon_color;
uniform vec3 sun_color;
uniform vec3 sun_direction;

out vec4 f_color;

const float SUN_RADIUS = 1.5;
const float SUN_HALO_POWER = 64.0;
const float SUN_HALO_STRENGTH = 0.25;

const float HORIZON_STRENGTH = 0.30;
const float HORIZON_WIDTH = 3.0;

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

    sky = mix(sky_main_color, sky_zenith_color, clamp(y, 0.0, 1.0));

    // warm atmosphere
    float horizon = exp(-abs(y) * HORIZON_WIDTH);

    sky += sky_horizon_color * horizon * HORIZON_STRENGTH;

    // sun disk
    float sun_dot = dot(ray, normalize(sun_direction));
    float sun_radius = radians(SUN_RADIUS);
    float sun_disk = smoothstep(
        cos(sun_radius),
        cos(sun_radius * 0.65),
        sun_dot
    );

    sky += sun_color * sun_disk;

    // sun halo
    float halo = pow(max(sun_dot, 0.0), SUN_HALO_POWER);

    sky+= sun_color * halo * SUN_HALO_STRENGTH;

    f_color = vec4(sky, 1.0);
}
