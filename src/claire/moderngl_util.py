from __future__ import annotations

import moderngl


def get_uniform(program: moderngl.Program, label: str) -> moderngl.Uniform:
    should_be_uniform = program[label]
    if not isinstance(should_be_uniform, moderngl.Uniform):
        msg = f"The uniform {label} is not a moderngl.Uniform"
        raise TypeError(msg)
    return should_be_uniform
