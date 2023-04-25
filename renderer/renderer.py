"""Vectorized implementations."""

from functools import partial
from typing import Callable, NewType, Optional, Union

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Integer, jaxtyped

from .geometry import compute_normals, to_homogeneous
from .types import (BatchCanvasMask, Canvas, CanvasMask, Colour, FaceIndices,
                    Index, Texture, Triangle, Triangle3D, TriangleBarycentric,
                    TriangleColours, Vec2i, Vec3f, Vertices, World2Screen,
                    ZBuffer)

jax.config.update('jax_array', True)


@jaxtyped
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

    def set_mask(mask: CanvasMask, x: Index, y: Index) -> CanvasMask:
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


@jaxtyped
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


@jaxtyped
@partial(jax.jit, donate_argnums=(1, 2))
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
    pts_zs: Vec3f = pts[:, 2]  # floats

    def compute_u(x: Index, y: Index) -> Vec3f:
        """Compute barycentric coordinate u."""
        return barycentric(pts_2d, jnp.array((x, y)))

    # barycentric coordinate for all pixels
    coordinates: Float[Array, "width height 3"] = jax.vmap(
        jax.vmap(compute_u))(
            *jnp.broadcast_arrays(
                # various x, along first axis
                lax.expand_dims(jnp.arange(jnp.size(canvas, 0)), [1]),
                # various y, along second axis
                lax.expand_dims(jnp.arange(jnp.size(canvas, 1)), [0]),
            ))

    def compute_z(coord: Float[Array, "3"]) -> Float[Array, ""]:
        """Compute z using barycentric coordinates."""
        return jnp.dot(coord, pts_zs)

    _zbuffer: ZBuffer = jax.vmap(jax.vmap(compute_z))(coordinates)

    visible_mask = jnp.logical_and(
        # z value >= existing marks (in `zbuffer`) are visible.
        _zbuffer >= zbuffer,
        # in / on triangle if barycentric coordinate is non-negative
        (coordinates >= 0).all(axis=2, keepdims=False),
    )

    # visible_mask: expand to colour channel dimension
    canvas = jnp.where(jnp.expand_dims(visible_mask, axis=2), colour, canvas)
    zbuffer = lax.select(visible_mask, _zbuffer, zbuffer)

    return zbuffer, canvas


@jaxtyped
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

    def compute_u(x: Index, y: Index) -> Vec3f:
        """Compute barycentric coordinate u."""
        return barycentric(pts_2d, jnp.array((x, y)))

    coordinates = jax.vmap(jax.vmap(compute_u))(
        *jnp.broadcast_arrays(
            # various x, along first axis
            lax.expand_dims(jnp.arange(jnp.size(canvas, 0)), [1]),
            # various y, along second axis
            lax.expand_dims(jnp.arange(jnp.size(canvas, 1)), [0]),
        ))

    def compute_z(coord: TriangleBarycentric) -> Vec3f:
        """Compute z using barycentric coordinates."""
        return jnp.dot(coord, pts_zs)

    # in / on triangle if barycentric coordinate is non-negative
    valid_coords = (coordinates >= 0).all(axis=2, keepdims=False)

    # set to 0 to reduce computations maybe?
    coordinates = jnp.where(jnp.expand_dims(valid_coords, 2), coordinates, 0)

    _zbuffer: ZBuffer = jax.vmap(jax.vmap(compute_z))(coordinates)
    # z value >= existing marks (in `zbuffer`) are visible.
    visible_mask = jnp.logical_and(_zbuffer >= zbuffer, valid_coords)

    def compute_colour(coord: TriangleBarycentric) -> TriangleColours:
        """Compute colours using barycentric coordinates."""
        return coord.dot(colours)

    _canvas: Canvas = jax.vmap(jax.vmap(compute_colour))(coordinates)

    # visible_mask: expand to colour channel dimension
    canvas = jnp.where(jnp.expand_dims(visible_mask, axis=2), _canvas, canvas)
    zbuffer = lax.select(visible_mask, _zbuffer, zbuffer)

    return zbuffer, canvas


_Colours = NewType(
    "_Colours",
    # either (3 ) for triangle3d or (3 3) for texture
    Union[Colour, TriangleColours],
)
_BatchColours = NewType(
    "_BatchColours",
    # either (3 ) for triangle3d or (3 3) for texture
    Union[Float[Array, "faces 3"], Float[Array, "faces 3 3"]],
)


@jaxtyped
@partial(jax.jit, static_argnames=("triangle_renderer"), donate_argnums=(2, 3))
def _batched_rendering(
    triangle_renderer: Callable[[Triangle3D, ZBuffer, Canvas, _Colours],
                                tuple[ZBuffer, Canvas]],
    screen_coords: Float[Array, "faces 3 3"],
    zbuffer: ZBuffer,
    canvas: Canvas,
    # either (faces 3) for triangle3d or (faces 3 3) for texture
    colours: _BatchColours,
) -> tuple[ZBuffer, Canvas]:
    faces: int = int(jnp.size(colours, 0))

    def _render_one_batch(
        i: Integer[Array, ""],
        carry: tuple[ZBuffer, Canvas],
    ) -> tuple[ZBuffer, Canvas]:
        _zbuffer, _canvas = carry

        # with back-face culling
        _zbuffer, _canvas = lax.cond(
            (colours[i] >= 0).all(),
            lambda __zbuffer, __canvas: triangle_renderer(
                screen_coords[i],
                __zbuffer,
                __canvas,
                colours[i],
            ),
            lambda __zbuffer, __canvas: (__zbuffer, __canvas),
            _zbuffer,
            _canvas,
        )

        return _zbuffer, _canvas

    zbuffer, canvas = lax.fori_loop(
        0,
        faces,
        _render_one_batch,
        (zbuffer, canvas),
    )

    return zbuffer, canvas


@jaxtyped
@partial(jax.jit, donate_argnums=(0, 1))
def _render_with_simple_light(
    canvas: Canvas,
    zbuffer: ZBuffer,
    screen_coords: Float[Array, "faces 3 3"],
    intensities: Float[Array, " faces"],
    light_colour: Colour,
) -> tuple[ZBuffer, Canvas]:
    colours: Float[Array, "faces channel"]
    colours = intensities[:, None] * light_colour[None, :]
    assert isinstance(colours, Float[Array, "faces channel"])

    zbuffer, canvas = _batched_rendering(
        triangle_renderer=triangle3d,
        screen_coords=screen_coords,
        zbuffer=zbuffer,
        canvas=canvas,
        colours=colours,
    )
    assert isinstance(zbuffer, ZBuffer)
    assert isinstance(canvas, Canvas)

    return zbuffer, canvas


@jaxtyped
@partial(jax.jit, donate_argnums=(0, 1))
def _render_with_texture(
    canvas: Canvas,
    zbuffer: ZBuffer,
    world_coords: Float[Array, "faces 3 3"],
    screen_coords: Float[Array, "faces 3 3"],
    intensities: Float[Array, " faces"],
    light_colour: Colour,
    texture: Texture,
) -> tuple[ZBuffer, Canvas]:
    assert texture is not None
    texture = lax.cond(
        texture.max() > 1.0,
        # normalise texture values to [0...1] if it is 8-bit [0...255]
        lambda: texture / 255.,
        lambda: texture,
    )

    world2texture: Float[Array, "3 2"] = (
        # 4. divide by half to center
        (jnp.identity(2) * 0.5)
        # 3. multiply by width, height to map to whole texture map
        @ (jnp.diag(texture.shape[:2]))
        # 2. homogeneous coordinate to cartesian x-y
        @ jnp.eye(2, 3)
        # 1. add 1 to ensure all coordinates are positive
        @ jnp.identity(3).at[:2, -1].set(1))

    texture_coords: Integer[Array, "faces 3 2"]
    # set "z" as 1 to 1) eliminate z dim, & 2) make it 2D homogeneous coords
    texture_coords = (world2texture @ world_coords.at[..., -1].set(1).swapaxes(
        1, 2)).swapaxes(1, 2).astype(int)
    assert isinstance(texture_coords, Integer[Array, "faces 3 2"])

    @jaxtyped
    @jax.jit
    def get_colour(texture_coord: Vec2i) -> Colour:
        return texture[texture_coord[0], texture_coord[1], :]

    # colours of each vertex
    colours: Float[Array, "faces 3 channel"]
    # first map along faces axis, then along 3 (3 verts per triangle)
    colours = jax.vmap(jax.vmap(get_colour))(texture_coords)
    colours = colours * intensities[:, None, None] * light_colour
    assert isinstance(colours, Float[Array, "faces 3 channel"])

    zbuffer, canvas = _batched_rendering(
        triangle_renderer=triangle_texture,
        screen_coords=screen_coords,
        zbuffer=zbuffer,
        canvas=canvas,
        colours=colours,
    )
    assert isinstance(zbuffer, ZBuffer)
    assert isinstance(canvas, Canvas)

    return zbuffer, canvas


@jaxtyped
@partial(jax.jit, donate_argnums=(0, 1))
def render(
    zbuffer: ZBuffer,
    canvas: Canvas,
    faces: FaceIndices,
    verts: Vertices,
    light_direction: Vec3f,
    light_colour: Colour,
    world2screen: World2Screen,
    texture: Optional[Texture] = None,
) -> tuple[ZBuffer, Canvas]:
    """Render triangulated object under simple light with/without texture.

    Noted that the rendered result will be a ZBuffer and a Canvas, both in
    (width, height, *) shape. To render them properly, the width and height
    dimensions may need to be swapped.

    Parameters:
      - zbuffer: ZBuffer, should be in shape of (width, height).
      - canvas: Canvas, should in shape of (width, height, channel), with
        `channel` be the same as the size of `light_colour`.
      - faces: FaceIndices (Integer[Array, "faces 3"]). Vertex indices of each
        triangle.
      - verts: Vertices (Float[Array, "vertices 3"]). Cartesian coordinates of
        each vertex.
      - light_direction: Vec3f. Vector indicating the direction of the light.
      - light_colour: Colour. Indicating the colour of light (that will be
        multiplied onto the intensity).
      - world2screen: World2Screen (Float[Array, "4 3"]). Used to convert model
        coordinates to screen/canvas coordinates. If not given, it assumes all
        model coordinates are in [-1...0] and will transform them into
        ([0...canvas width], [0...canvas height], [-1...0]).
      - texture: Optional[Texture]

    Returns: tuple[ZBuffer, Canvas]
      - ZBuffer: Num[Array, "width height"], same as `zbuffer` given.
      - Canvas: Num[Array, "width height channel"], same as `canvas` given.
    """
    assert jnp.ndim(light_direction) == 1, "light direction must be 1D vector"

    coord_dtype = jax.dtypes.result_type(zbuffer)
    # faces, verts per face, x-y-z
    world_coords: Float[Array, "faces 3 3"] = verts[faces].astype(coord_dtype)

    screen_coords: Float[Array, "faces 3 3"] = (
        world2screen @ to_homogeneous(world_coords).swapaxes(1, 2)).swapaxes(
            1, 2)
    # ensure coords are at the actual pixels
    screen_coords = screen_coords.at[..., :2].set(
        lax.floor(screen_coords.at[..., :2].get()))
    assert isinstance(screen_coords, Float[Array, "faces 3 3"])

    normals: Float[Array, "faces 3"] = compute_normals(world_coords)
    assert isinstance(normals, Float[Array, "faces 3"])

    intensities: Float[Array, " faces"] = jnp.dot(normals, light_direction)

    if texture is None:
        return _render_with_simple_light(
            canvas=canvas,
            zbuffer=zbuffer,
            screen_coords=screen_coords,
            intensities=intensities,
            light_colour=light_colour,
        )

    return _render_with_texture(
        canvas=canvas,
        zbuffer=zbuffer,
        world_coords=world_coords,
        screen_coords=screen_coords,
        intensities=intensities,
        light_colour=light_colour,
        texture=texture,
    )
