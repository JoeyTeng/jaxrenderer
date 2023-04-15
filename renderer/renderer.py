"""Vectorized implementations."""

import jax
import jax.numpy as jnp
from jax import lax

from .types import (BatchCanvasMask, Canvas, CanvasMask, Colour, Triangle,
                    Triangle3D, TriangleColours, Vec2i, Vec3f, ZBuffer)

jax.config.update('jax_array', True)


@jax.jit
def line(
    t0: Vec2i,
    t1: Vec2i,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    """Paint a line using `colour` onto `canvas`. Returns the updated `canvas`.

    Parameters:
      - t0, t1: jax array, int, shape (2, ) (x-y dimension)
      - canvas: jax array, float, shape (w, h, c) (w, h, 3 colour channels)
      - colour: jax array, float, shape (c, ) (3 colour channels)
    """
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


@jax.jit
def barycentric(pts: Triangle, p: Vec2i) -> Vec3f:
    """Compute the barycentric coordinate of `p`.
        Returns u[-1] < 0 if `p` is outside of the triangle.
    """
    mat: jax.Array = jnp.vstack((
        pts[2] - pts[0],
        pts[1] - pts[0],
        pts[0] - p,
    ))
    vec: Vec3f = jnp.cross(mat[:, 0], mat[:, 1]).astype(float)
    # `pts` and `P` has integer value as coordinates so `abs(u[2])` < 1 means
    # `u[2]` is 0, that means triangle is degenerate, in this case
    # return something with negative coordinates
    vec = lax.cond(
        jnp.abs(vec[-1]) < 1,
        lambda: jnp.array((-1., 1., 1.)),
        lambda: vec,
    )
    vec = vec / vec[-1]
    vec = jnp.array((1 - (vec[0] + vec[1]), vec[1], vec[0]))

    return vec


@jax.jit
def triangle3d(
    pts: Triangle3D,
    zbuffer: ZBuffer,
    canvas: Canvas,
    colour: Colour,
) -> tuple[ZBuffer, Canvas]:
    """Paint a triangle using `colour`, respect `zbuffer` onto `canvas`.
        Returns the updated `zbuffer` and `canvas`.

    Parameters:
      - pts: jax array, float, shape (3, 3) (number of points, x-y-z dimension)
      - zbuffer: jax array, float, shape (w, h)
      - canvas: jax array, float, shape (w, h, c) (w, h, 3 colour channels)
      - colour: jax array, float, shape (c, ) (3 colour channels)
    """
    pts_2d: Triangle = pts[:, :2].astype(int)
    pts_zs = pts[:, 2]  # floats

    def compute_u(x: int, y: int) -> Vec3f:
        """Compute barycentric coordinate u."""
        return barycentric(pts_2d, jnp.array((x, y)))

    coordinates = jax.vmap(jax.vmap(compute_u, (0, 0), 0), (0, 0), 0)(
        *jnp.broadcast_arrays(
            # various x, along first axis
            lax.expand_dims(jnp.arange(jnp.size(canvas, 0)), [1]),
            # various y, along second axis
            lax.expand_dims(jnp.arange(jnp.size(canvas, 1)), [0]),
        ))

    def f(coord: jax.Array) -> jax.Array:
        """Compute z using barycentric coordinates."""
        return jnp.dot(coord, pts_zs)

    # in / on triangle if barycentric coordinate is non-negative
    valid_coords = jnp.where(jnp.less_equal(0, coordinates), True, False)
    valid_coords = valid_coords.all(axis=2, keepdims=False)

    _zbuffer: ZBuffer = jax.vmap(jax.vmap(f, (0, ), 0), (0, ), 0)(coordinates)
    # z value >= existing marks (in `zbuffer`) are visible.
    visible_mask = jnp.where(jnp.greater_equal(_zbuffer, zbuffer), True, False)
    visible_mask = jnp.logical_and(visible_mask, valid_coords)

    # visible_mask: expand to colour channel dimension
    canvas = jnp.where(jnp.expand_dims(visible_mask, axis=2), colour, canvas)
    zbuffer = jnp.where(visible_mask, _zbuffer, zbuffer)

    return zbuffer, canvas


@jax.jit
def triangle_texture(
    pts: Triangle3D,
    zbuffer: ZBuffer,
    canvas: Canvas,
    colours: TriangleColours,
) -> tuple[ZBuffer, Canvas]:
    """Paint a triangle using `colours`, respect `zbuffer` onto `canvas`.
        The colour painted onto each pixel is interpolated using 3 colours
        in vertices. Returns the updated `zbuffer` and `canvas`.

    Parameters:
      - pts: jax array, float, shape (3, 3) (number of points, x-y-z dimension)
      - zbuffer: jax array, float, shape (w, h)
      - canvas: jax array, float, shape (w, h, c) (w, h, 3 colour channels)
      - colours: jax array, float, shape (c, ) (3 colour channels)
    """
    pts_2d: Triangle = pts[:, :2].astype(int)
    pts_zs = pts[:, 2]  # floats

    def compute_u(x: int, y: int) -> Vec3f:
        """Compute barycentric coordinate u."""
        return barycentric(pts_2d, jnp.array((x, y)))

    coordinates = jax.vmap(jax.vmap(compute_u, (0, 0), 0), (0, 0), 0)(
        *jnp.broadcast_arrays(
            # various x, along first axis
            lax.expand_dims(jnp.arange(jnp.size(canvas, 0)), [1]),
            # various y, along second axis
            lax.expand_dims(jnp.arange(jnp.size(canvas, 1)), [0]),
        ))

    def compute_z(coord: jax.Array) -> jax.Array:
        """Compute z using barycentric coordinates."""
        return jnp.dot(coord, pts_zs)

    # in / on triangle if barycentric coordinate is non-negative
    valid_coords = jnp.where(jnp.less_equal(0, coordinates), True, False)
    valid_coords = valid_coords.all(axis=2, keepdims=False)

    # set to 0 to reduce computations maybe?
    coordinates = jnp.where(jnp.expand_dims(valid_coords, 2), coordinates, 0)

    _zbuffer: ZBuffer = jax.vmap(jax.vmap(compute_z))(coordinates)
    # z value >= existing marks (in `zbuffer`) are visible.
    visible_mask = jnp.where(jnp.greater_equal(_zbuffer, zbuffer), True, False)
    visible_mask = jnp.logical_and(visible_mask, valid_coords)

    def compute_colour(coord: jax.Array) -> jax.Array:
        """Compute colours using barycentric coordinates."""
        return coord.dot(colours)

    _canvas: Canvas = jax.vmap(jax.vmap(compute_colour))(coordinates)

    # visible_mask: expand to colour channel dimension
    canvas = jnp.where(jnp.expand_dims(visible_mask, axis=2), _canvas, canvas)
    zbuffer = jnp.where(visible_mask, _zbuffer, zbuffer)

    return zbuffer, canvas
