"""Vectorized implementations."""

from typing import NewType, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_array', True)

# CanvasMask = NewType("CanvasMask", jax.Array)
CanvasMask = Sequence[Sequence[bool]]
# BatchCanvasMask = NewType("CanvasMask", jax.Array)
BatchCanvasMask = Sequence[Sequence[Sequence[bool]]]
# Canvas = NewType("Canvas", jax.Array)
Canvas = Sequence[Sequence[Sequence[float]]]
# ZBuffer = NewType("ZBuffer", jax.Array)
ZBuffer = Sequence[Sequence[float]]
# Colour = NewType("Colour", jax.Array)
Colour = Tuple[float, float, float]
# TriangleColours = NewType("TriangleColours", jax.Array)
TriangleColours = Tuple[Colour, Colour, Colour]
# Vec2i = NewType("Vec2i", jax.Array)
Vec2i = Tuple[int, int]
# Vec3f = NewType("Vec3f", jax.Array)
Vec3f = Tuple[float, float, float]
# Triangle = NewType("Triangle", jax.Array)
Triangle = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
# Triangle3D = NewType("Triangle3D", jax.Array)
Triangle3D = Tuple[Tuple[float, float, float], Tuple[float, float, float],
                   Tuple[float, float, float]]


@jax.jit
def line(
    t0: Vec2i,
    t1: Vec2i,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    x0, y0, x1, y1 = t0[0], t0[1], t1[0], t1[1]

    def set_mask(mask: CanvasMask, x: int, y: int) -> CanvasMask:
        return mask.at[x, y].set(True)

    batch_set_mask = jax.vmap(set_mask, in_axes=0, out_axes=0)

    row_idx = jnp.arange(jnp.size(canvas, 0))
    col_idx = jnp.arange(jnp.size(canvas, 1))

    # where mask. Final result would be a rectangle of `True`s.
    where_row_mask: CanvasMask = jnp.logical_and(
        lax.min(x0, x1) < row_idx,
        row_idx < lax.max(x0, x1),
    ).reshape((-1, 1))
    where_col_mask: CanvasMask = jnp.logical_and(
        lax.min(y0, y1) < col_idx,
        col_idx < lax.max(y0, y1),
    ).reshape((1, -1))
    # where mask. Broadcast and use "and" to take intersection
    where_mask: CanvasMask = jnp.logical_and(
        *jnp.broadcast_arrays(where_row_mask, where_col_mask))

    # scan along horizontal
    xs = row_idx
    ts = (xs - x0) / jnp.array(x1 - x0, float)
    ys = (y0 * (1. - ts) + y1 * ts).astype(jnp.int32)

    # horizontal
    batch_mask: BatchCanvasMask = lax.full(
        (jnp.size(canvas, 0), jnp.size(canvas, 0), jnp.size(canvas, 1)),
        False,
        dtype=jnp.bool_.dtype,
    )
    # mapped
    batch_mask = batch_set_mask(batch_mask, xs, ys)
    mask = batch_mask.any(axis=0, keepdims=True, where=where_mask)

    # scan along vertical
    ys = col_idx
    ts = (ys - y0) / jnp.array(y1 - y0, float)
    xs = (x0 * (1. - ts) + x1 * ts).astype(jnp.int32)

    # vertical
    batch_mask = jnp.broadcast_to(
        mask, (jnp.size(canvas, 1), jnp.size(canvas, 0), jnp.size(canvas, 1)))
    batch_mask = batch_set_mask(batch_mask, xs, ys)
    mask = batch_mask.any(axis=0, keepdims=False, where=where_mask)

    # where mask is True, using `colour`; otherwise keep canvas value
    canvas = jnp.where(lax.expand_dims(mask, [2]), colour, canvas)

    return canvas
