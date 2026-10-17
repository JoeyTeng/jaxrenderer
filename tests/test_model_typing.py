"""Focused runtime checks for model array validation and merging."""

from typing import Any

from jax import Array
import jax.numpy as jnp
from jaxtyping import TypeCheckError
import pytest

from renderer._meta_utils import typed_asarray
from renderer.model import MergedModel, Model
from renderer.types import FaceIndices, Normals, Texture, UVCoordinates, Vertices


def _array(values: object, dtype: Any) -> Array:
    return typed_asarray(values, dtype=dtype)


def _model_fields(
    verts: Vertices,
) -> tuple[Vertices, Normals, UVCoordinates, FaceIndices, Texture]:
    return (
        verts,
        _array(((1.0, 0.0, 0.0),) * 3, jnp.float32),
        _array(((0.0, 0.0),) * 3, jnp.float32),
        _array(((0, 1, 2),), jnp.int32),
        _array(((1.0, 1.0, 1.0),) * 6, jnp.float32).reshape((2, 3, 3)),
    )


def test_model_create_rejects_vertex_dtype_and_shape_mismatches() -> None:
    with pytest.raises((AssertionError, TypeCheckError)):
        _ = Model.create(*_model_fields(_array(((1, 2, 3),) * 3, jnp.int32)))

    with pytest.raises((AssertionError, TypeCheckError)):
        _ = Model.create(*_model_fields(_array(((1.0, 2.0),) * 3, jnp.float32)))


def test_merge_maps_preserves_padding_and_returns_map_dimensions() -> None:
    first = _array(((1.0, 2.0),), jnp.float32)
    second = _array(((3.0,), (4.0,)), jnp.float32)

    merged, (width, height) = MergedModel.merge_maps((first, second))

    assert (width, height) == (2, 2)
    assert merged.shape == (4, 2)
    assert jnp.array_equal(
        merged,
        _array(((1.0, 2.0), (0.0, 0.0), (3.0, 0.0), (4.0, 0.0)), jnp.float32),
    )


def test_generate_object_vert_info_broadcasts_vector_values_per_vertex() -> None:
    generated = MergedModel.generate_object_vert_info(
        (2, 1),
        (_array((1.0, 2.0), jnp.float32), _array((3.0, 4.0), jnp.float32)),
    )

    assert generated.shape == (3, 2)
    assert jnp.array_equal(
        generated,
        _array(((1.0, 2.0), (1.0, 2.0), (3.0, 4.0)), jnp.float32),
    )
