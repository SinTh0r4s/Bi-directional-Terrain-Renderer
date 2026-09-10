from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyglm import glm

from claire.terrain.lod_selector import LodData, LodSelector, create_load_data

if TYPE_CHECKING:
    from collections.abc import Iterator

    from claire.aabb import AABB
    from claire.terrain.numpy_types import HeightmapData


@dataclass
class Node:
    lod_data: LodData
    terrain_offset: glm.ivec2
    lod_level: int
    children: list[Node]


class ChunkSelection:
    def __init__(self, max_lod_level: int, mesh_size: int) -> None:
        self._max_lod_level = max_lod_level
        self._mesh_size = mesh_size
        self._chunk_map: dict[tuple[int, int, int], Node] = {}

    def add(self, node: Node) -> None:
        self._chunk_map[node.lod_level, node.terrain_offset.x, node.terrain_offset.y] = node

    def has_refinements_of(self, node: Node) -> bool:
        child_stride = 1 << (node.lod_level - 1)
        return (
            (node.lod_level - 1, node.terrain_offset.x, node.terrain_offset.y) in self._chunk_map
            or (node.lod_level - 1, node.terrain_offset.x, node.terrain_offset.y + child_stride) in self._chunk_map
            or (node.lod_level - 1, node.terrain_offset.x + child_stride, node.terrain_offset.y) in self._chunk_map
            or (node.lod_level - 1, node.terrain_offset.x + child_stride, node.terrain_offset.y + child_stride)
            in self._chunk_map
        )

    def __iter__(self) -> Iterator[Node]:
        return iter(self._chunk_map.values())

    def _sample_neighbor_lod(self, requesting_lod_level: int, x: int, y: int) -> int:
        for lod_level in range(requesting_lod_level, self._max_lod_level + 1):
            stride = 1 << lod_level
            tile_size = self._mesh_size * stride
            x = (x // tile_size) * tile_size
            y = (y // tile_size) * tile_size
            if (lod_level, x, y) in self._chunk_map:
                return lod_level
        return requesting_lod_level

    def get_neighbor_lods_nwse(self, node: Node) -> glm.ivec4:
        tile_size = (1 << node.lod_level) * self._mesh_size
        return glm.ivec4(
            self._sample_neighbor_lod(node.lod_level, node.terrain_offset.x, node.terrain_offset.y - tile_size),
            self._sample_neighbor_lod(node.lod_level, node.terrain_offset.x - tile_size, node.terrain_offset.y),
            self._sample_neighbor_lod(node.lod_level, node.terrain_offset.x, node.terrain_offset.y + tile_size),
            self._sample_neighbor_lod(node.lod_level, node.terrain_offset.x + tile_size, node.terrain_offset.y),
        )


def _create_node(
    heightmap: HeightmapData, terrain_offset: glm.ivec2, tile_size: int, current_lod_level: int
) -> Node | None:
    lod_stride = 1 << current_lod_level
    lod_data = create_load_data(heightmap, terrain_offset, lod_stride, tile_size)
    if lod_data is None:
        return None
    if current_lod_level == 0:
        return Node(lod_data, terrain_offset, current_lod_level, [])
    lod_stride >>= 1
    delta_offset = lod_stride * tile_size
    children = [
        _create_node(heightmap, terrain_offset, tile_size, current_lod_level - 1),
        _create_node(heightmap, terrain_offset + glm.ivec2(0, delta_offset), tile_size, current_lod_level - 1),
        _create_node(heightmap, terrain_offset + glm.ivec2(delta_offset, 0), tile_size, current_lod_level - 1),
        _create_node(
            heightmap, terrain_offset + glm.ivec2(delta_offset, delta_offset), tile_size, current_lod_level - 1
        ),
    ]
    return Node(lod_data, terrain_offset, current_lod_level, [child for child in children if child is not None])


def _create_root_nodes(heightmap: HeightmapData, mesh_size_exponent: int, max_lod_level: int) -> list[Node]:
    root_nodes: list[Node | None] = []
    mesh_size = 1 << mesh_size_exponent
    root_chunk_size = 1 << (mesh_size_exponent + max_lod_level)
    cols, rows = heightmap.shape
    for chunk_col in range(math.ceil(cols / root_chunk_size)):
        for chunk_row in range(math.ceil(rows / root_chunk_size)):
            offset = glm.ivec2(chunk_col, chunk_row) * root_chunk_size
            root_nodes.append(_create_node(heightmap, offset, mesh_size, max_lod_level))
    return [node for node in root_nodes if node is not None]


class QuadTree:
    def __init__(self, heightmap: HeightmapData, mesh_size_exponent: int, max_lod_level: int) -> None:
        self._root_nodes = _create_root_nodes(heightmap, mesh_size_exponent, max_lod_level)
        self._previous_selection: ChunkSelection | None = None
        self._max_lod_level = max_lod_level
        self._mesh_size = 1 << mesh_size_exponent

    def filter(self, lod_selector: LodSelector[LodData]) -> ChunkSelection:
        selection = ChunkSelection(self._max_lod_level, self._mesh_size)

        def filter_node(node: Node) -> None:
            state = lod_selector.should_refine(node.lod_data)
            if state == "use_previous":
                state = (
                    "render"
                    if self._previous_selection is None
                    else ("refine" if self._previous_selection.has_refinements_of(node) else "render")
                )
            if state == "render" or len(node.children) == 0:
                selection.add(node)
            else:  # state == "refine"
                for child in node.children:
                    filter_node(child)

        for node in self._root_nodes:
            filter_node(node)
        self._previous_selection = selection
        return selection

    def get_aabbs(self) -> list[AABB]:
        if self._previous_selection is None:
            return []
        return [chunk.lod_data.aabb for chunk in self._previous_selection]
