from typing import NewType, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_array', True)

# Canvas = NewType("Canvas", jax.Array)
Canvas = Sequence[Sequence[Sequence[int]]]
# Colour = NewType("Colour", jax.Array)
Colour = Sequence[int]


@jax.jit
def line(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    steep: bool = lax.abs(x0 - x1) < lax.abs(y0 - y1)
    # if steep, swap x y
    x0, y0, x1, y1 = lax.cond(
        steep,
        lambda: (y0, x0, y1, x1),
        lambda: (x0, y0, x1, y1),
    )
    # if x0 - x1 < 0, swap point 0 and 1
    x0, x1, y0, y1 = lax.cond(
        x0 > x1,
        lambda: (x1, x0, y1, y0),
        lambda: (x0, x1, y0, y1),
    )

    dx: int = x1 - x0
    dy: int = y1 - y0
    incY: int = lax.cond(y1 > y0, lambda: 1, lambda: -1)
    derror2: int = lax.abs(dy) * 2

    def f(x: int, state: Tuple[int, int, Canvas]) -> Tuple[int, int, Canvas]:
        y, error2, _canvas = state
        _canvas = lax.cond(
            steep,
            lambda: _canvas.at[y, x, :].set(colour),
            lambda: _canvas.at[x, y, :].set(colour),
        )

        error2 += derror2
        y, error2 = lax.cond(
            error2 > dx,
            lambda: (y + incY, error2 - dx * 2),
            lambda: (y, error2),
        )

        return (y, error2, _canvas)

    _, __, canvas = lax.fori_loop(x0, (x1 + 1), f, (y0, 0, canvas))

    return canvas
