from pathlib import Path
from typing import Final

import moderngl
import numpy as np
from skimage.io import imread

from claire.camera import Camera

_VERTEX_SHADER: Final[Path] = Path(__file__).parent / "vertex_shader.glsl"
_FRAGMENT_SHADER: Final[Path] = Path(__file__).parent / "fragment_shader.glsl"


def _cut_into_textures(heightmap: np.ndarray, size: int) -> np.ndarray:
    width, height = heightmap.shape
    stride = size - 1
    layers = []
    for x_start in range(0, width, stride):
        for y_start in range(0, height, stride):
            x_end = min(x_start + size, width)
            y_end = min(y_start + size, height)
            tile = heightmap[x_start:x_end, y_start:y_end]
            padded = np.empty((size, size), dtype=heightmap.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            layers.append(padded)
    return np.stack(layers)


class DEM:
    def __init__(self, heightmap_tif: Path) -> None:
        self._heightmap = imread(heightmap_tif)
        self._heightmap[self._heightmap < 0] = 0
        self._width, self._height = self._heightmap.shape
        if self._heightmap.dtype != np.float32:
            msg = "Requiring a heightmap of float32!"
            raise ValueError(msg)

    def upload(self, ctx: moderngl.Context) -> None:
        try:
            self._program = ctx.program(
                vertex_shader=_VERTEX_SHADER.read_text(encoding="utf-8"),
                fragment_shader=_FRAGMENT_SHADER.read_text(encoding="utf-8")
            )
        except Exception as e:
            print(e)
            raise
        texture_size = 1024
        layers = _cut_into_textures(self._heightmap, size=1024)
        self._heightmap_texture = ctx.texture_array(
            (texture_size, texture_size, len(layers)),
            components=1,
            data=layers.tobytes(),
            dtype="f4",
        )
        self._vao = ctx.vertex_array(self._program, [])

    def render(self, camera: Camera) -> None:
        self._heightmap_texture.use(location=0)
        self._program["heightmap"] = 0
        self._program["num_columns"] = self._width
        self._program["offset_x"] = 0.0
        self._program["offset_y"] = 0.0
        self._program["lod_stride"] = 1
        mvp = camera.proj_matrix() * camera.view_matrix()
        self._program["mvp"].write(mvp.to_bytes())

        rows_of_squares = self._height - 1
        vertices_per_row_of_squares = self._width * 2
        degenerate_vertices = (rows_of_squares - 1)  # one between every row of squares
        vertex_count = rows_of_squares * vertices_per_row_of_squares + degenerate_vertices
        self._vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=vertex_count)
