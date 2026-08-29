import math

from pyglm import glm


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


def _initial_yaw_and_pitch(camera_position: glm.vec3, look_at: glm.vec3) -> tuple[float, float]:
    direction = glm.normalize(look_at - camera_position)
    yaw = glm.degrees(glm.atan(direction.x, -direction.z))
    pitch = glm.degrees(glm.asin(direction.y))
    return yaw, pitch


class Camera:
    def __init__(self, start_position: glm.vec3, look_at: glm.vec3):
        self.position = start_position
        self.yaw, self.pitch = _initial_yaw_and_pitch(start_position, look_at)

        self._move_speed = 10.0
        self._rotation_speed = 0.15

        self._left_active = False
        self._right_active = False
        self._up_active = False
        self._down_active = False
        self._forward_active = False
        self._backward_active = False

    def rotate(self, horizontal: float, vertical: float) -> None:
        self.yaw += horizontal * self._rotation_speed
        self.pitch += -vertical * self._rotation_speed
        self.pitch = glm.clamp(self.pitch, -89.0, 89.0)

    def start_left(self) -> None:
        self._left_active = True

    def stop_left(self) -> None:
        self._left_active = False

    def start_right(self) -> None:
        self._right_active = True

    def stop_right(self) -> None:
        self._right_active = False

    def start_up(self) -> None:
        self._up_active = True

    def stop_up(self) -> None:
        self._up_active = False

    def start_down(self) -> None:
        self._down_active = True

    def stop_down(self) -> None:
        self._down_active = False

    def start_forward(self) -> None:
        self._forward_active = True

    def stop_forward(self) -> None:
        self._forward_active = False

    def start_backward(self) -> None:
        self._backward_active = True

    def stop_backward(self) -> None:
        self._backward_active = False

    def on_update(self, delta_time: float) -> None:
        forward = self._forward_vector()
        right = self._right_vector()
        up = glm.cross(forward, right)

        if self._left_active:
            self.position += -right * self._move_speed * delta_time
        if self._right_active:
            self.position += right * self._move_speed * delta_time

        if self._up_active:
            self.position += up * self._move_speed * delta_time
        if self._down_active:
            self.position += -up * self._move_speed * delta_time

        if self._forward_active:
            self.position += forward * self._move_speed * delta_time
        if self._backward_active:
            self.position += -forward * self._move_speed * delta_time

    def _forward_vector(self) -> glm.vec3:
        yaw = glm.radians(self.yaw)
        pitch = glm.radians(self.pitch)
        return glm.normalize(glm.vec3(
            glm.sin(yaw) * glm.cos(pitch),
            glm.sin(pitch),
            -glm.cos(yaw) * glm.cos(pitch),
        ))

    def _right_vector(self) -> glm.vec3:
        return glm.normalize(
            glm.cross(
                self._forward_vector(),
                glm.vec3(0.0, 1.0, 0.0),
            )
        )

    def view_matrix(self) -> glm.mat4:
        return glm.lookAt(
            self.position,
            self.position + self._forward_vector(),
            glm.vec3(0.0, 1.0, 0.0),
        )

    def proj_matrix(self) -> glm.mat4:
        return glm.perspective(math.radians(45.0), 4.0 / 3.0, 1.0, 25_000.0)

    def is_visible(self, aabb_min: glm.vec3, aabb_max: glm.vec3) -> bool:
        planes = _extract_frustum_planes(self.view_matrix())
        return _is_aabb_in_frustum(planes, aabb_min, aabb_max)

    def min_distance_to_camera(self, aabb_min: glm.vec3, aabb_max: glm.vec3) -> float:
        return glm.length(
            glm.vec3(
                glm.clamp(self.position.x, aabb_min.x, aabb_max.x),
                glm.clamp(self.position.y, aabb_min.y, aabb_max.y),
                glm.clamp(self.position.z, aabb_min.z, aabb_max.z)
            )
        )