#version 330

uniform sampler2D image;
uniform vec3 color_bias;
uniform float image_over_viewport_aspect_ratio;
uniform bool is_greyscale;
uniform float blend_alpha;

in vec2 uv;

out vec4 f_color;

void main() {
    float scale_x = min(image_over_viewport_aspect_ratio, 1.0);
    float scale_y = min(1.0 / image_over_viewport_aspect_ratio, 1.0);
    vec2 image_uv = vec2((uv.x - 0.5) / scale_x + 0.5, (uv.y - 0.5) / scale_y + 0.5);
    f_color = texture(image, image_uv);
    f_color.rgb = is_greyscale ? f_color.rrr : f_color.rgb;
    f_color.a *= float(int(image_uv.x >= 0.0 && image_uv.x <= 1.0 && image_uv.y >= 0.0 && image_uv.y <= 1.0));
    f_color *= vec4(color_bias, blend_alpha);
}
