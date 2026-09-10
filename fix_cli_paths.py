"""
Printed paths by ruff and basedpyright are often not recognized by PyCharm. This fixes the syntax to ensure links can be
clicked for convenience
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable


def _uv_run_command(command: str, transform_log: Callable[[str], str], *arguments: str) -> NoReturn:
    cli_command = ["uv", "run", command, *arguments]
    print(f"Poe => {' '.join(cli_command)}")  # noqa: T201
    process = subprocess.run(cli_command, capture_output=True)  # noqa: PLW1510, S603
    for line in process.stderr.splitlines():
        print(transform_log(line.decode("utf-8")), file=sys.stderr)  # noqa: T201
    for line in process.stdout.splitlines():
        print(transform_log(line.decode("utf-8")))  # noqa: T201
    sys.exit(process.returncode)


def ruff_with_output_format_full(mandatory_arguments: list[str], *arguments: str) -> NoReturn:
    cwd_prefix = Path.cwd().resolve().as_uri()

    def path_to_uri(line: str) -> str:
        if "--> " in line:
            line = line.replace("\\", "/").replace("--> ", f"  {cwd_prefix}/")
        return line

    if len(arguments) == 0:
        arguments = tuple(".")
    _uv_run_command("ruff", path_to_uri, *mandatory_arguments, "--output-format=full", *arguments)


def basedpyright(*arguments: str) -> NoReturn:
    regex = re.compile(r"^\s\s[a-zA-Z]:\\")

    def path_to_uri(line: str) -> str:
        if regex.match(line):
            line = "  file:///" + line[2:].replace("\\", "/")
        return line

    _uv_run_command("basedpyright", path_to_uri, *arguments)
