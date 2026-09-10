from __future__ import annotations

from typing import TYPE_CHECKING

from pyglm import glm

if TYPE_CHECKING:
    from claire.camera import HasCameraPosition, HasViewProjMatrices


class AABB:
    """Axis-aligned bounding box"""

    def __init__(self, position_min: glm.vec3, position_max: glm.vec3) -> None:
        self._position_min = position_min
        self._position_max = position_max

    def get_unit_cube_model_matrix(self) -> glm.mat4:
        return glm.translate(self._position_min) * glm.scale(self._position_max - self._position_min)

    def get_shortest_distance_to(self, camera: HasCameraPosition) -> float:
        return glm.length(
            glm.vec3(
                glm.clamp(camera.position.x, self._position_min.x, self._position_max.x),
                glm.clamp(camera.position.y, self._position_min.y, self._position_max.y),
                glm.clamp(camera.position.z, self._position_min.z, self._position_max.z),
            )
            - camera.position
        )

    def is_visible(self, camera: HasViewProjMatrices) -> bool:
        view_proj = camera.proj_matrix() * camera.view_matrix()
        # Each plane is represented as:
        # ax + by + cz + d >= 0  -> inside
        #
        # glm matrices are column-major, so extracting rows explicitly
        # gives the correct clip-space inequalities.
        planes: tuple[glm.vec4, ...] = (
            # Left:   x + w >= 0  # noqa: ERA001
            glm.vec4(
                view_proj[0][3] + view_proj[0][0],
                view_proj[1][3] + view_proj[1][0],
                view_proj[2][3] + view_proj[2][0],
                view_proj[3][3] + view_proj[3][0],
            ),
            # Right:  w - x >= 0  # noqa: ERA001
            glm.vec4(
                view_proj[0][3] - view_proj[0][0],
                view_proj[1][3] - view_proj[1][0],
                view_proj[2][3] - view_proj[2][0],
                view_proj[3][3] - view_proj[3][0],
            ),
            # Bottom: y + w >= 0  # noqa: ERA001
            glm.vec4(
                view_proj[0][3] + view_proj[0][1],
                view_proj[1][3] + view_proj[1][1],
                view_proj[2][3] + view_proj[2][1],
                view_proj[3][3] + view_proj[3][1],
            ),
            # Top:    w - y >= 0  # noqa: ERA001
            glm.vec4(
                view_proj[0][3] - view_proj[0][1],
                view_proj[1][3] - view_proj[1][1],
                view_proj[2][3] - view_proj[2][1],
                view_proj[3][3] - view_proj[3][1],
            ),
            # Near:   z + w >= 0  (OpenGL)  # noqa: ERA001
            glm.vec4(
                view_proj[0][3] + view_proj[0][2],
                view_proj[1][3] + view_proj[1][2],
                view_proj[2][3] + view_proj[2][2],
                view_proj[3][3] + view_proj[3][2],
            ),
            # Far:    w - z >= 0  # noqa: ERA001
            glm.vec4(
                view_proj[0][3] - view_proj[0][2],
                view_proj[1][3] - view_proj[1][2],
                view_proj[2][3] - view_proj[2][2],
                view_proj[3][3] - view_proj[3][2],
            ),
        )

        for plane in planes:
            positive: glm.vec3 = glm.vec3(
                self._position_max.x if plane.x >= 0.0 else self._position_min.x,
                self._position_max.y if plane.y >= 0.0 else self._position_min.y,
                self._position_max.z if plane.z >= 0.0 else self._position_min.z,
            )
            if (plane.x * positive.x + plane.y * positive.y + plane.z * positive.z + plane.w) < 0.0:
                return False
        return True
