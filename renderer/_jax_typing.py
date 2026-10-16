"""Narrow, manually maintained typing aliases for JAX calls used here.

These aliases describe the practical subset used by this project. The casts do
not validate upstream signatures, and ``Array`` does not prove shape or dtype.
``grad`` supports one ``Array`` argument, a scalar-``Array`` loss and default
options; its type does not guarantee a particular shape. Check this subset
against JAX when upgrading the dependency.
"""

from collections.abc import Callable
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import jax.typing

type _Shape = int | tuple[int, ...]


class _ArrayFactory(Protocol):
    def __call__(
        self, value: object, /, dtype: jax.typing.DTypeLike | None = None
    ) -> jax.Array: ...


class _ShapeFactory(Protocol):
    def __call__(
        self, shape: _Shape, dtype: jax.typing.DTypeLike | None = None
    ) -> jax.Array: ...


class _FullFactory(Protocol):
    def __call__(
        self,
        shape: _Shape,
        fill_value: jax.typing.ArrayLike,
        dtype: jax.typing.DTypeLike | None = None,
    ) -> jax.Array: ...


class _Modf(Protocol):
    def __call__(self, x: jax.Array, /) -> tuple[jax.Array, jax.Array]: ...


class _Grad(Protocol):
    def __call__(
        self, fun: Callable[[jax.Array], jax.Array], /
    ) -> Callable[[jax.Array], jax.Array]: ...


class _JaxAPI(Protocol):
    grad: _Grad


class _JaxNumpyAPI(Protocol):
    array: _ArrayFactory
    asarray: _ArrayFactory
    ones: _ShapeFactory
    zeros: _ShapeFactory
    full: _FullFactory
    modf: _Modf


# These casts declare the subset used by this project; they do not wrap JAX.
_jax_api = cast(_JaxAPI, jax)
_jnp_api = cast(_JaxNumpyAPI, jnp)

array = _jnp_api.array
asarray = _jnp_api.asarray
ones = _jnp_api.ones
zeros = _jnp_api.zeros
full = _jnp_api.full
modf = _jnp_api.modf
grad = _jax_api.grad
