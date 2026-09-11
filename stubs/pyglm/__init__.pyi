"""Match the outdated glm type stubs to match the new import"""
from __future__ import annotations

import glm as _glm

# Re-export it so 'from pyglm import glm' resolves perfectly
glm = _glm
