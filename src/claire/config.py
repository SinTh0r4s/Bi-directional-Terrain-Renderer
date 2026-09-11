from __future__ import annotations

from dataclasses import dataclass, field

from .lighting import Lighting
from .terrain.lod_selector import LodConfig
from .terrain_picture_overlay import TerrainPictureOverlayConfig


@dataclass
class Config:
    """All these settings may be changed at any time to change all future renderings"""

    lighting: Lighting = field(default_factory=Lighting)
    picture_overlay: TerrainPictureOverlayConfig = field(default_factory=TerrainPictureOverlayConfig)
    lod_config: LodConfig = field(default_factory=LodConfig)
