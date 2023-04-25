from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import jaxtyped, Array, Integer, Num, Shaped

from .types import Canvas, Colour, ZBuffer


@jaxtyped
@partial(jax.jit, static_argnames=("canvas_size", ))
def create_buffer(
    canvas_size: tuple[int, int],
    background_colour: Colour,
    canvas_dtype: Optional[jnp.dtype] = None,
    zbuffer_dtype: jnp.dtype = jnp.single,
) -> tuple[ZBuffer, Canvas]:
    """Render triangulated object under simple light with/without texture.

    Noted that the rendered result will be a ZBuffer and a Canvas, both in
    (width, height, *) shape. To render them properly, the width and height
    dimensions may need to be swapped.

    Parameters:
      - `canvas_size`: tuple[int, int]. Width, height of the resultant image.
      - `background_colour`: Optional[Colour]. Used to fill the canvas before
        anything being rendered. If not given (or None), using
        `jnp.zeros(canvas_size, dtype=canvas_dtype)`, which will resulting in
        a black background.
      - `canvas_dtype`: dtype for canvas. If not given, the dtype of the
        `background_colour` will be used.
      - `zbuffer_dtype`: dtype for canvas. Default: `jnp.single`.

    Returns: tuple[ZBuffer, Canvas]
      - ZBuffer: Num[Array, "width height"], with dtype being the same as
        `zbuffer_dtype` or `jnp.single` if not given.
      - Canvas: Num[Array, "width height channel"], with dtype being the same
        as the given one, or `background_colour`. "channel" is given by the size
        of `background_colour`.
    """
    width, height = canvas_size
    channel: int = background_colour.size
    canvas_dtype = (jax.dtypes.result_type(background_colour)
                    if canvas_dtype is None else canvas_dtype)

    canvas: Canvas = jnp.full(
        (width, height, channel),
        background_colour,
        dtype=canvas_dtype,
    )
    min_z = (jnp.finfo(dtype=zbuffer_dtype).min if jnp.issubdtype(
        zbuffer_dtype, jnp.floating) else jnp.iinfo(dtype=zbuffer_dtype))
    zbuffer: ZBuffer = jnp.full(
        canvas_size,
        min_z,
        dtype=zbuffer_dtype,
    )

    return zbuffer, canvas


@jaxtyped
@jax.jit
def get_value_from_index(
    matrix: Shaped[Array, "width height batch *valueDimensions"],
    index: Integer[Array, "width height"],
) -> Shaped[Array, "width height *valueDimensions"]:
    """Retrieve value along 3rd axis using index value from index matrix."""
    return jax.vmap(jax.vmap(lambda mt, ix: mt[ix]))(matrix, index)


@jaxtyped
@jax.jit
def merge_canvases(
    zbuffers: Num[Array, "batch width height"],
    canvases: Shaped[Array, "batch width height channel"],
) -> tuple[ZBuffer, Canvas]:
    """Merge canvases by selecting each pixel with max z value in zbuffer,
        then merge zbuffer as well.
    """
    pixel_idx: Integer[Array, "width height"] = jnp.argmax(zbuffers, axis=0)
    assert isinstance(pixel_idx, Integer[Array, "width height"])

    zbuffer: ZBuffer = get_value_from_index(
        lax.transpose(zbuffers, (1, 2, 0)),
        pixel_idx,
    )
    assert isinstance(zbuffer, ZBuffer)

    canvas: Canvas = get_value_from_index(
        # first vmap along width, then height, then choose among "faces"
        lax.transpose(canvases, (1, 2, 0, 3)),
        pixel_idx,
    )
    assert isinstance(canvas, Canvas)

    return zbuffer, canvas


@jaxtyped
@jax.jit
def transpose_for_display(
        matrix: Num[Array,
                    "fst snd *channel"]) -> Num[Array, "snd fst *channel"]:
    return jnp.swapaxes(matrix, 0, 1)
