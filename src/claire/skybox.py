from __future__ import annotations

from typing import TYPE_CHECKING

from pyglm import glm

from .fullscreen_quad import FullscreenQuad

if TYPE_CHECKING:
    import moderngl

    from .camera import Camera
    from .lighting import Lighting


class Skybox:
    def __init__(self, ctx: moderngl.Context) -> None:
        self._quad = FullscreenQuad(ctx, "skybox")

    def render(self, camera: Camera, lighting: Lighting) -> None:
        self._quad.get_uniform("inv_proj").write(glm.inverse(camera.proj_matrix()).to_bytes())
        self._quad.get_uniform("inv_view").write(glm.inverse(camera.view_matrix()).to_bytes())
        self._quad.get_uniform("viewport_width_height").write(camera.resolution.to_bytes())
        self._quad.get_uniform("sky_zenith_color").write(lighting.sky_zenith_color.to_bytes())
        self._quad.get_uniform("sky_main_color").write(lighting.sky_main_color.to_bytes())
        self._quad.get_uniform("sky_horizon_color").write(lighting.sky_horizon_color.to_bytes())
        self._quad.get_uniform("sun_color").write(lighting.sun_color.to_bytes())
        self._quad.get_uniform("sun_direction").write(lighting.sun_direction.to_bytes())
        self._quad.render()
