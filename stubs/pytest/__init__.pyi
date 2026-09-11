
from typing import Any

from _pytest.fixtures import fixture as fixture  # Re-export fixture safely
from _pytest.python_api import ApproxBase

def approx(
    expected: Any,
    rel: float | None = None,
    abs: float | None = None,
    nan_ok: bool = False,
) -> ApproxBase: ...
