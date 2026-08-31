from pyglm import glm

from claire.camera import Camera


def _extract_frustum_planes(view_projection: glm.mat4) -> list[glm.vec4]:
    """
    Extracts the 6 frustum planes from a column-major View-Projection matrix
    using the Gribb-Hartmann method.
    Planes are returned as glm.vec4(A, B, C, D) where Ax + By + Cz + D = 0.
    """
    # PyGLM matrix access is m[column][row]
    col1 = view_projection[0]
    col2 = view_projection[1]
    col3 = view_projection[2]
    col4 = view_projection[3]

    planes = [
        col4 + col1,  # Left
        col4 - col1,  # Right
        col4 + col2,  # Bottom
        col4 - col2,  # Top
        col4 + col3,  # Near
        col4 - col3  # Far
    ]

    # Normalize the planes so the length of the normal (A, B, C) is 1
    for i in range(6):
        normal = glm.vec3(planes[i])
        length = glm.length(normal)
        if length > 0.0:
            planes[i] /= length

    return planes


def _is_aabb_in_frustum(planes: list[glm.vec4], aabb_min: glm.vec3, aabb_max: glm.vec3) -> bool:
    """
    Tests an AABB against the 6 Gribb-Hartmann planes.
    Returns True if visible or intersecting, False if completely culled.
    """
    for plane in planes:
        # Find the positive vertex (the corner furthest along the plane normal)
        p = glm.vec3(
            aabb_max.x if plane.x >= 0 else aabb_min.x,
            aabb_max.y if plane.y >= 0 else aabb_min.y,
            aabb_max.z if plane.z >= 0 else aabb_min.z
        )

        # Test if the positive vertex is behind the plane (outside the frustum)
        # Ax + By + Cz + D < 0
        if glm.dot(glm.vec3(plane), p) + plane.w < 0:
            return False  # Completely outside this plane, cull it!

    return True


class AABB:
    """Axis-aligned bounding box"""
    def __init__(self, position_min: glm.vec3, position_max: glm.vec3) -> None:
        self._position_min = position_min
        self._position_max = position_max

    def get_shortest_distance_to(self, camera: Camera) -> float:
        return glm.length(
            glm.vec3(
                glm.clamp(camera.position.x, self._position_min.x, self._position_max.x),
                glm.clamp(camera.position.y, self._position_min.y, self._position_max.y),
                glm.clamp(camera.position.z, self._position_min.z, self._position_max.z)
            )
        )

    def is_visible(self, camera: Camera) -> bool:
        planes = _extract_frustum_planes(camera.view_matrix())
        return _is_aabb_in_frustum(planes, self._position_min, self._position_max)
