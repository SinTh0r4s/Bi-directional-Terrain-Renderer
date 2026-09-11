from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import moderngl

    from claire.camera import Camera


class TerrainPicture:
    def __init__(
        self, ctx: moderngl.Context, width: int, height: int, image: bytes, image_format: Literal["greyscale", "color"]
    ) -> None:
        self._texture = ctx.texture(
            size=(width, height), components=1 if image_format == "greyscale" else 4, data=image
        )
        self._width = width
        self._height = height
        self._is_greyscale = image_format == "greyscale"
        self._camera: Camera | None = None

    def __del__(self) -> None:
        self._texture.release()

    @property
    def aspect_ratio(self) -> float:
        return self._width / self._height

    @property
    def is_greyscale(self) -> bool:
        return self._is_greyscale

    @property
    def camera(self) -> Camera | None:
        return self._camera

    @camera.setter
    def camera(self, camera: Camera) -> None:
        self._camera = camera.copy()
        self._camera.resolution.x = self._width
        self._camera.resolution.y = self._height

    def bind_to_location(self, location: int) -> None:
        self._texture.use(location=location)
