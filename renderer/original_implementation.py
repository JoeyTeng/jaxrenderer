import jax
import jax.numpy as jnp
from jax import lax

from .types import (Canvas, Colour, Triangle, Triangle3D, TriangleColours,
                    Vec2i, Vec3f, ZBuffer)

jax.config.update('jax_array', True)


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

    def f(x: int, state: tuple[int, int, Canvas]) -> tuple[int, int, Canvas]:
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
def barycentric(pts: Triangle, p: Vec2i) -> Vec3f:
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
    pts_2d: Triangle = pts[:, :2].astype(int)
    # min_x, min_y
    mins: Vec2i = lax.clamp(
        0,
        jnp.min(pts_2d, axis=0),
        jnp.array(canvas.shape[:2]),
    )
    # max_x, max_y
    maxs: Vec2i = lax.clamp(
        0,
        jnp.max(pts_2d, axis=0),
        jnp.array(canvas.shape[:2]),
    )
    pts_zs = pts[:, 2]  # floats

    def g(
        y: int,
        state: tuple[int, ZBuffer, Canvas],
    ) -> tuple[int, ZBuffer, Canvas]:
        x: int
        _zbuffer: ZBuffer
        _canvas: Canvas
        x, _zbuffer, _canvas = state

        coord: Vec3f = barycentric(pts_2d, jnp.array((x, y)))
        z: float = jnp.dot(coord, pts_zs)

        _zbuffer, _canvas = lax.cond(
            jnp.concatenate((
                jnp.array([_zbuffer[x, y] > z]),
                jnp.less(coord, 0.),
            )).any(),
            lambda: (_zbuffer, _canvas),
            lambda: (
                _zbuffer.at[x, y].set(z),
                _canvas.at[x, y, :].set(colour),
            ),
        )

        return x, _zbuffer, _canvas

    def f(x: int, state: tuple[ZBuffer, Canvas]) -> tuple[ZBuffer, Canvas]:
        _zbuffer: ZBuffer
        _canvas: Canvas
        _zbuffer, _canvas = state

        _, _zbuffer, _canvas = lax.fori_loop(
            mins[1],
            maxs[1] + 1,
            g,
            (x, _zbuffer, _canvas),
        )

        return _zbuffer, _canvas

    zbuffer, canvas = lax.fori_loop(mins[0], maxs[0] + 1, f, (zbuffer, canvas))

    return zbuffer, canvas


@jax.jit
def triangle_texture(
    pts: Triangle3D,
    zbuffer: ZBuffer,
    canvas: Canvas,
    colours: TriangleColours,
) -> tuple[ZBuffer, Canvas]:
    pts_2d: Triangle = pts[:, :2].astype(int)
    # min_x, min_y
    mins: Vec2i = lax.clamp(
        0,
        jnp.min(pts_2d, axis=0),
        jnp.array(canvas.shape[:2]),
    )
    # max_x, max_y
    maxs: Vec2i = lax.clamp(
        0,
        jnp.max(pts_2d, axis=0),
        jnp.array(canvas.shape[:2]),
    )
    pts_zs = pts[:, 2]  # floats

    def g(
        y: int,
        state: tuple[int, ZBuffer, Canvas],
    ) -> tuple[int, ZBuffer, Canvas]:
        x: int
        _zbuffer: ZBuffer
        _canvas: Canvas
        x, _zbuffer, _canvas = state

        coord: Vec3f = barycentric(pts_2d, jnp.array((x, y)))
        z: float = jnp.dot(coord, pts_zs)

        _zbuffer, _canvas = lax.cond(
            jnp.concatenate((
                jnp.array([_zbuffer[x, y] > z]),
                jnp.less(coord, 0.),
            )).any(),
            lambda: (_zbuffer, _canvas),
            lambda: (
                _zbuffer.at[x, y].set(z),
                # weighted sum of colour; not commutative.
                _canvas.at[x, y, :].set(coord.dot(colours)),
            ),
        )

        return x, _zbuffer, _canvas

    def f(x: int, state: tuple[ZBuffer, Canvas]) -> tuple[ZBuffer, Canvas]:
        _zbuffer: ZBuffer
        _canvas: Canvas
        _zbuffer, _canvas = state

        _, _zbuffer, _canvas = lax.fori_loop(
            mins[1],
            maxs[1] + 1,
            g,
            (x, _zbuffer, _canvas),
        )

        return _zbuffer, _canvas

    zbuffer, canvas = lax.fori_loop(mins[0], maxs[0] + 1, f, (zbuffer, canvas))

    return zbuffer, canvas


@jax.jit
def triangle(
    pts: Triangle,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    # min_x, min_y
    mins: Vec2i = lax.clamp(
        0,
        jnp.min(pts, axis=0),
        jnp.array(canvas.shape[:2]),
    )
    # max_x, max_y
    maxs: Vec2i = lax.clamp(
        0,
        jnp.max(pts, axis=0),
        jnp.array(canvas.shape[:2]),
    )

    def g(y: int, state: tuple[int, Canvas]) -> tuple[int, Canvas]:
        x: int
        _canvas: Canvas
        x, _canvas = state
        coord: Vec3f = barycentric(pts, jnp.array((x, y)))
        _canvas = lax.cond(
            jnp.less(coord, 0.).any(),
            lambda: _canvas,
            lambda: _canvas.at[x, y, :].set(colour),
        )

        return x, _canvas

    def f(x: int, _canvas: Canvas) -> Canvas:
        _canvas = lax.fori_loop(mins[1], maxs[1] + 1, g, (x, _canvas))[1]

        return _canvas

    canvas = lax.fori_loop(mins[0], maxs[0] + 1, f, canvas)

    return canvas


@jax.jit
def triangle_sweep(
    pts: Triangle,
    canvas: Canvas,
    colour: Colour,
) -> Canvas:
    """Line sweeping."""
    t0: Vec2i = pts[0]
    t1: Vec2i = pts[1]
    t2: Vec2i = pts[2]

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

    def fill_colour(x: int, state: tuple[int, Canvas]) -> tuple[int, Canvas]:
        _canvas: Canvas
        y, _canvas = state
        _canvas = _canvas.at[x, y, :].set(colour)

        return y, _canvas

    def draw_row(
        y: int,
        state: tuple[Vec2i, int, int, Canvas],
    ) -> tuple[Vec2i, int, int, Canvas]:
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
