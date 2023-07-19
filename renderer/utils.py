from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import Sequence, Union, cast

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Num, Shaped
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import Tuple
from ._meta_utils import add_tracing_name
from ._meta_utils import typed_jit as jit
from .types import Canvas, IntV, Texture, ZBuffer


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def get_value_from_index(
    matrix: Shaped[Array, "width height batch *valueDimensions"],
    index: Integer[Array, "width height"],
) -> Shaped[Array, "width height *valueDimensions"]:
    """Retrieve value along 3rd axis using index value from index matrix."""

    def _get(
        mt: Shaped[Array, "batch *valueDimensions"],
        ix: IntV,
    ) -> Shaped[Array, "*valueDimensions"]:
        return mt[ix]

    return jax.vmap(jax.vmap(_get))(matrix, index)


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def merge_canvases(
    zbuffers: Num[Array, "batch width height"],
    canvases: Shaped[Array, "batch width height channel"],
) -> Tuple[ZBuffer, Canvas]:
    """Merge canvases by selecting each pixel with max z value in zbuffer,
    then merge zbuffer as well.
    """
    pixel_idx: Integer[Array, "width height"]
    pixel_idx = jnp.argmax(zbuffers, axis=0)  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(pixel_idx, Integer[Array, "width height"])

    zbuffer: ZBuffer = get_value_from_index(
        lax.transpose(  # pyright: ignore[reportUnknownMemberType]
            zbuffers,
            (1, 2, 0),
        ),
        pixel_idx,
    )
    assert isinstance(zbuffer, ZBuffer)

    canvas: Canvas = get_value_from_index(
        # first vmap along width, then height, then choose among "faces"
        lax.transpose(  # pyright: ignore[reportUnknownMemberType]
            canvases,
            (1, 2, 0, 3),
        ),
        pixel_idx,
    )
    assert isinstance(canvas, Canvas)

    return zbuffer, canvas


@jaxtyped
@partial(
    jit,
    inline=True,
    static_argnames=("flip_vertical",),
)
@add_tracing_name
def transpose_for_display(
    matrix: Num[Array, "fst snd *channel"],
    flip_vertical: bool = True,
) -> Num[Array, "snd fst *channel"]:
    """Transpose matrix for display.

    When flip_vertical is disabled, the matrix's origin ([0, 0]) is assumed to
    be at bottom-left. Thus, the correct way to display the matrix is to using
    tools like matplotlib is to specify `origin="lower"`.
    To be compatible with PyTinyrenderer and most image processing programs,
    the default behaviour is to flip vertically.
    """
    mat = cast(Num[Array, "snd fst *channel"], jnp.swapaxes(matrix, 0, 1))
    assert isinstance(mat, Num[Array, "snd fst *channel"])
    if flip_vertical:
        mat = mat[::-1, ...]

    return mat


@jaxtyped
@add_tracing_name
def build_texture_from_PyTinyrenderer(
    texture: Union[Num[Array, "length"], Sequence[float]],
    width: int,
    height: int,
) -> Texture:
    """Build a texture from PyTinyrenderer's format.

    The texture was specified in C order (channel varies the fastest), but with
    y as the first axis. Besides, after swapping the first two axes, the second axis is reversed as required by this renderer.

    Parameters:
      - texture: a 1D array of length `width * height * channels`, where each
        channel elements represent a pixel in RGB order. When channels is 1,
        the resulted texture still has 3 dimensions, with last dimension of
        side 1.
      - width: width of the texture.
      - height: height of the texture.

    Returns: A texture with shape `(width, height, channels)`.
    """
    return jnp.reshape(  # pyright: ignore[reportUnknownMemberType]
        jnp.asarray(texture),  # pyright: ignore[reportUnknownMemberType]
        (width, height, -1),
        order="C",
    ).swapaxes(0, 1)[:, ::-1, :]
