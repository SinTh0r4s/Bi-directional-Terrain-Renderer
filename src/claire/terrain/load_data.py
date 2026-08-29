from dataclasses import dataclass
from pathlib import Path
from skimage.io import imread

import numpy as np


@dataclass
class DataChunk:
    heightmap: np.ndarray
    latitude_left: float
    longitude_bottom: float


def load_geodata(tif: Path, chunk_size: int = 1024) -> list[DataChunk]:
    geodata = imread(tif)
    geodata[geodata < 0] = 0
    width, height = geodata.shape
    chunks: list[DataChunk] = []
    step_size = chunk_size - 1
    for row in range(0, width, step_size):
        for column in range(0, height, step_size):
            r_end = min(row + chunk_size, width)
            c_end = min(column + chunk_size, height)
            if r_end - row <= 1 or c_end - column <= 1:
                continue
            chunk = geodata[row:r_end, column:c_end]
            chunks.append(
                DataChunk(
                    heightmap=chunk,
                    latitude_left=row,
                    longitude_bottom=column
                )
            )
    return chunks
