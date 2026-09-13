from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Literal

import moderngl

_assets: Final[Path] = Path(__file__).parent

_include_regex: re.Pattern[str] = re.compile(r'^\s*#include\s+["<](?P<name>[^">\s]+)[">][ \t]*$', re.MULTILINE)
_bake_regex: re.Pattern[str] = re.compile(r'^\s*#bake\s+(?P<name>[^">\s]+)[ \t]*$', re.MULTILINE)
_file_id_regex: re.Pattern[str] = re.compile(r"(?P<file_id>\d+)[:(](?P<line>\d+)\)?", re.MULTILINE)
_version_regex: re.Pattern[str] = re.compile(r"^(?P<version>\s*#version\s+\d+[ \t]*)$", re.MULTILINE)


def _read_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


class BakeConstant:
    def __init__(self, name: str, data_type: str, value: str | float) -> None:
        self.name = name
        self._data_type = data_type
        self._value = str(value)

    def __str__(self) -> str:
        return f"const {self._data_type} {self.name} = {self._value};"


class _ShaderSource:
    def __init__(
        self,
        shader_type: Literal["vert", "frag"],
        file_id_start: int,
        name: str,
        bake_constants: list[BakeConstant] | None = None,
    ) -> None:
        self._shader_type = shader_type
        self.name = name
        self.bake_constants = {constant.name: constant for constant in bake_constants or []}
        self._filename = f"{self.name}.{self._shader_type}.glsl"
        self._file_id_start = file_id_start
        self._imports: dict[str, int] = {self._filename: file_id_start}

    def _include_match_handler(self, match: re.Match[str]) -> str:
        filename = match.group("name")
        if filename in self._imports:
            file_id = self._imports[filename]
        else:
            file_id = len(self._imports) + self._file_id_start
            self._imports[filename] = file_id
        current_line = match.string.count("\n", 0, match.start())
        return (
            f"#line {0} {file_id}\n"
            + _read_safe(_assets / filename)
            + f"#line {current_line + 1} {self._file_id_start}"
        )

    def _bake_match_handler(self, match: re.Match[str]) -> str:
        key = match.group("name")
        if key not in self.bake_constants:
            msg = f"No bake target found in {self._filename} for constant '{key}'"
            raise ValueError(msg)
        return str(self.bake_constants[key])

    def _version_match_handler(self, match: re.Match[str]) -> str:
        version = match.group("version")
        return f"{version}\n#line 1 {self._file_id_start}"

    def load(self) -> tuple[str, dict[int, str]]:
        source = _read_safe(_assets / self._filename)
        source = _version_regex.sub(self._version_match_handler, source)
        source = _include_regex.sub(self._include_match_handler, source)
        source = _bake_regex.sub(self._bake_match_handler, source)
        return source, {file_id: filename for filename, file_id in self._imports.items()}


def load_program(
    ctx: moderngl.Context,
    vert_or_both: str | tuple[str, list[BakeConstant]],
    frag: str | tuple[str, list[BakeConstant]] | None = None,
) -> moderngl.Program:
    match vert_or_both:
        case str():
            vert_source = _ShaderSource("vert", 0, vert_or_both)
        case tuple():
            vert_source = _ShaderSource("vert", 0, *vert_or_both)
    match frag:
        case str():
            frag_source = _ShaderSource("frag", 1024, frag)
        case tuple():
            frag_source = _ShaderSource("frag", 1024, *frag)
        case None:
            frag_source = _ShaderSource("frag", 1024, vert_source.name, list(vert_source.bake_constants.values()))
    vert_resolved, vert_imports = vert_source.load()
    frag_resolved, frag_imports = frag_source.load()
    try:
        return ctx.program(vertex_shader=vert_resolved, fragment_shader=frag_resolved)
    except moderngl.Error as e:
        import_name_map: dict[int, str] = {}
        import_name_map.update(vert_imports)
        import_name_map.update(frag_imports)

        def replace_file_ids(match: re.Match[str]) -> str:
            filename = import_name_map[int(match.group("file_id"))]
            line = match.group("line")
            return f"{filename}:{line}"

        message = _file_id_regex.sub(replace_file_ids, str(e))
        raise moderngl.Error(message) from None
