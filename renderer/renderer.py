from typing import NewType, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_array', True)

# Canvas = NewType("Canvas", jax.Array)
Canvas = Sequence[Sequence[Sequence[int]]]
# Colour = NewType("Colour", jax.Array)
Colour = Sequence[int]
# Vec2i = NewType("Vec2i", jax.Array)
Vec2i = Tuple[int, int]


@jax.jit
def line(
    t0: Vec2i,
    t1: Vec2i,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    x0, y0, x1, y1 = t0[0], t0[1], t1[0], t1[1]

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


@jax.jit
def triangle(
    t0: Vec2i,
    t1: Vec2i,
    t2: Vec2i,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    """Line sweeping."""
    # sort vertices by ascending y-coordinates
    t0, t1 = lax.cond(t0[1] < t1[1], lambda: (t0, t1), lambda: (t1, t0))
    t0, t2 = lax.cond(t0[1] < t2[1], lambda: (t0, t2), lambda: (t2, t0))
    t1, t2 = lax.cond(t1[1] < t2[1], lambda: (t1, t2), lambda: (t2, t1))

    # changes of y or x between t{} and t{}, dy >= 0
    dy02: int = t2[1] - t0[1]
    dy01: int = lax.max(t1[1] - t0[1], 1)  # to prevent div by zero
    dy12: int = lax.max(t2[1] - t1[1], 1)  # to prevent div by zero
    dx02: int = t2[0] - t0[0]
    dx01: int = t1[0] - t0[0]
    dx12: int = t2[0] - t1[0]
    # rasterize contour
    canvas = line(t0, t1, canvas, colour)
    canvas = line(t1, t2, canvas, colour)
    canvas = line(t0, t2, canvas, colour)

    def fill_colour(x: int, state: Tuple[int, Canvas]) -> Tuple[int, Canvas]:
        _canvas: Canvas
        y, _canvas = state
        _canvas = _canvas.at[x, y, :].set(colour)

        return y, _canvas

    def draw_row(
        y: int,
        state: Tuple[Vec2i, int, int, Canvas],
    ) -> Tuple[Vec2i, int, int, Canvas]:
        t: Vec2i
        dx: int
        dy: int
        _canvas: Canvas
        t, dx, dy, _canvas = state

        x02: int = t0[0] + dx02 * (y - t0[1]) // dy02
        x: int = t[0] + dx * (y - t[1]) // dy
        left: int
        right: int
        left, right = lax.cond(
            x02 < x,
            lambda: (x02, x),
            lambda: (x, x02),
        )

        _canvas = lax.fori_loop(left, right + 1, fill_colour, (y, _canvas))[-1]

        return t, dx, dy, _canvas

    # scan line
    canvas = lax.fori_loop(
        t0[1],
        t1[1] + 1,
        draw_row,
        (t0, dx01, dy01, canvas),
    )[-1]
    canvas = lax.fori_loop(
        t1[1],
        t2[1] + 1,
        draw_row,
        (t1, dx12, dy12, canvas),
    )[-1]

    return canvas
