from typing import Sequence, Union

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Integer, Num, Shaped, jaxtyped

from .types import Canvas, Texture, ZBuffer


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
    matrix: Num[Array, "fst snd *channel"],
    flip_vertical: bool = True,
) -> Num[Array, "snd fst *channel"]:
    """Transpose matrix for display.

    When flip_vertical is disabled, the matrix's origin ([0, 0]) is assumed to
    be at bottom-left. Thus, the correct way to display the matrix is to using
    tools like matplotlib is to specify `origin="lower"`.
    To be compatible with PyTinyrenderer and most image processing programs,
    the default behavior is to flip vertically.
    """
    mat = jnp.swapaxes(matrix, 0, 1)
    assert isinstance(mat, Num[Array, "snd fst *channel"])
    if flip_vertical:
        mat = mat[::-1, ...]

    return mat


@jaxtyped
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
    return jnp.reshape(
        jnp.asarray(texture),
        (width, height, -1),
        order="C",
    ).swapaxes(0, 1)[:, ::-1, :]
