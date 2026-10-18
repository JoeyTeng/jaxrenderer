"""Runtime checks for jaxtyping array specifications in geometry."""

from __future__ import annotations

from typing import Any

import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import (
    Float,
)
import pytest

from renderer._meta_utils import is_array_type, typed_ones
from renderer._meta_utils import typed_jaxtyped as jaxtyped


def _array(shape: tuple[int, ...], dtype: Any) -> Array:
    return typed_ones(shape, dtype=dtype)


@jaxtyped(typechecker=None)
def _matches_shared_vector_size(left: object, right: object) -> bool:
    return is_array_type(left, Float[Array, "n 3"]) and is_array_type(
        right, Float[Array, "n 3"]
    )


@jaxtyped(typechecker=None)
def _matches_float_matrix(value: object) -> bool:
    return is_array_type(value, Float[Array, "n 3"])


@jaxtyped(typechecker=None)
def _assert_float_matrix(value: object) -> bool:
    assert _matches_float_matrix(value)
    return True


def test_array_spec_check_rejects_mismatched_dtype_and_shape() -> None:
    value = _array((2, 3), jnp.float32)

    assert _matches_float_matrix(value)
    assert not _matches_float_matrix(_array((2, 3), jnp.int32))
    assert not _matches_float_matrix(_array((2, 4), jnp.float32))


def test_array_spec_check_binds_shared_dimensions_in_one_context() -> None:
    left = _array((2, 3), jnp.float32)
    same_size = _array((2, 3), jnp.float32)
    different_size = _array((4, 3), jnp.float32)

    assert _matches_shared_vector_size(left, same_size)
    assert not _matches_shared_vector_size(left, different_size)


def test_array_spec_check_rejects_dtype_and_shape_mismatches_during_jit_trace() -> None:
    compiled = jax.jit(_assert_float_matrix)  # pyright: ignore[reportUnknownMemberType]

    assert compiled(_array((2, 3), jnp.float32))
    with pytest.raises(AssertionError):
        compiled(_array((2, 3), jnp.int32))
    with pytest.raises(AssertionError):
        compiled(_array((2, 4), jnp.float32))
