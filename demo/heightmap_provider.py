from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
from skimage.io import imread

if TYPE_CHECKING:
    from claire.terrain.numpy_types import HeightmapData

_URL: Final[str] = (
    "https://olrc2.scholarsportal.info/dataverse/10.5683/SP3/PR368H/192fede4903-1ac7a5ee04a6?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27DSM_1m_UTM11N.tif&response-content-type=image%2Ftiff&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260828T164348Z&X-Amz-SignedHeaders=host&X-Amz-Credential=33b141c798354e21a3394e9e4f546bbe%2F20260828%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Expires=3600&X-Amz-Signature=b06f264292a874129c292a18ac9cd5c11a2749abf6eb73bcfde5069f5f82af79"
)
_PATH: Final[Path] = Path(__file__).parent / "DSM_1m_UTM11N.tif"


def load_heightmap() -> HeightmapData:
    if not _PATH.exists():
        print("Please open https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/PR368H")  # noqa: T201
        print(f"And download {_PATH.name} from page 2")  # noqa: T201
        print(f"Save it under {_PATH}")  # noqa: T201
        sys.exit(0)
    heightmap = cast("HeightmapData", imread(_PATH))
    if heightmap.dtype != np.float32:
        msg = "Requiring a heightmap of float32!"
        raise ValueError(msg)
    heightmap[heightmap <= 0] = 0
    return heightmap
