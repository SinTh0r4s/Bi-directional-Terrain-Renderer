from dataclasses import dataclass

from pyglm import glm


@dataclass
class Lighting:
    sky_zenith_color: glm.vec3 = glm.vec3(0.075, 0.15, 0.30)
    sky_main_color: glm.vec3 = glm.vec3(0.22, 0.39, 0.60)
    sky_horizon_color: glm.vec3 = glm.vec3(0.78, 0.74, 0.65)
    terrain_default_color: glm.vec3 = glm.vec3(0.68, 0.68, 0.68)
    sun_direction: glm.vec3 = glm.vec3(-0.6, 0.8, -0.35)
    sun_color: glm.vec3 = glm.vec3(1.0, 0.93, 0.82)
