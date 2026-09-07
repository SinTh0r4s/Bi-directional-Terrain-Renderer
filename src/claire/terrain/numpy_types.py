from typing_extensions import TypeAlias

import numpy as np

HeightmapData: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]