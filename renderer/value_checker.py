from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Integer, jaxtyped


@jaxtyped
@partial(jax.jit, inline=True)
def index_in_bound(
    indices: Integer[Array, "*any"],
    bound: Union[Integer[Array, ""], Integer[Array, "2"]],
) -> Bool[Array, ""]:
    """Check if indices are in bound.

    Parameters:
      - indices: indices to check, in any shape.
      - bound: bound to check against, assumed to be [min, max)
        (half-open interval) or [0, max) if only one value is given.
    """
    bound = jnp.asarray(bound).flatten()
    _min: int
    _max: int
    if bound.size == 2:
        _min, _max = bound
    else:
        _min, _max = 0, bound[0]

    return jnp.logical_and(
        (indices >= _min).all(),
        (indices < _max).all(),
    )
