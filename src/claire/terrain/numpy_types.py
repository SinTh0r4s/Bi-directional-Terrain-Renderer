from __future__ import annotations

import numpy as np
from typing_extensions import TypeAlias

HeightmapData: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
