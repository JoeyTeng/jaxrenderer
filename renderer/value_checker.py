from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Integer
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from .types import BoolV, IntV


@jaxtyped
def index_in_bound(
    indices: Integer[Array, "*any"],
    bound: Union[int, IntV, Integer[Array, "2"]],
) -> BoolV:
    """Check if indices are in bound.

    Parameters:
      - indices: indices to check, in any shape.
      - bound: bound to check against, assumed to be [min, max)
        (half-open interval) or [0, max) if only one value is given.
    """
    bound = jnp.asarray(bound).flatten()  # pyright: ignore[reportUnknownMemberType]
    _min: Union[int, IntV]
    _max: Union[int, IntV]
    if bound.size == 2:
        _min, _max = bound
    else:
        _min, _max = 0, bound[0]

    return jnp.logical_and(  # pyright: ignore[reportUnknownMemberType]
        (indices >= _min).all(),  # pyright: ignore[reportUnknownMemberType]
        (indices < _max).all(),  # pyright: ignore[reportUnknownMemberType]
    )
