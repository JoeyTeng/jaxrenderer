import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import jaxtyped, Array, Integer, Num, Shaped

from .types import Canvas, ZBuffer


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
