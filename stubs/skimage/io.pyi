
from pathlib import Path
from typing import Any

import numpy.typing as npt

def imread(
    fname: str | Path | Any, as_gray: bool = False, plugin: str | None = None, **kwargs: Any
) -> npt.NDArray[Any]: ...
