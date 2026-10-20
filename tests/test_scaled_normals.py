"""Regression checks for normals under per-object local scaling."""

from jax import Array
import jax.numpy as jnp
import pytest

from renderer._meta_utils import typed_asarray
from renderer.geometry import normalise
from renderer.model import MergedModel, Model, ModelObject, merge_objects
from renderer.types import FaceIndices, Normals, Texture, UVCoordinates, Vertices


def _float_array(values: object) -> Array:
    return typed_asarray(values, dtype=jnp.float32)


@pytest.mark.parametrize(
    "local_scaling",
    ((2.0, 0.5, 3.0), (-2.0, 0.5, 3.0)),
)
def test_merge_objects_transforms_scaled_normals_perpendicular_to_tangents(
    local_scaling: tuple[float, float, float],
) -> None:
    tangent_u = _float_array((1.0, 1.0, 0.0))
    tangent_v = _float_array((0.0, 1.0, 1.0))
    source_normal = jnp.cross(tangent_u, tangent_v)
    verts: Vertices = _float_array(((0.0, 0.0, 0.0), tangent_u, tangent_v))
    norms: Normals = jnp.broadcast_to(source_normal, (3, 3))
    uvs: UVCoordinates = _float_array(((0.0, 0.0),) * 3)
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=jnp.int32)
    diffuse_map: Texture = _float_array((((1.0, 1.0, 1.0),),))
    model = Model.create(verts, norms, uvs, faces, diffuse_map)

    rotation = _float_array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    transform = jnp.identity(4, dtype=jnp.float32)
    transform = transform.at[:3, :3].set(rotation)
    transform = transform.at[:3, 3].set((4.0, -2.0, 7.0))
    obj = ModelObject(
        model=model,
        local_scaling=_float_array(local_scaling),
        transform=transform,
    )

    merged: MergedModel = merge_objects((obj,))

    transformed_tangents: Array = jnp.stack(
        (merged.verts[1] - merged.verts[0], merged.verts[2] - merged.verts[0])
    )
    normal: Array = merged.norms[0]
    assert jnp.allclose(transformed_tangents @ normal, 0.0, atol=1e-6)

    # The cross product carries the tangent transform's determinant sign,
    # while inverse-transpose normals do not.
    expected: Array = jnp.cross(transformed_tangents[0], transformed_tangents[1])
    expected = expected * jnp.sign(jnp.prod(_float_array(local_scaling)))
    expected = normalise(expected)
    normal_direction = normalise(normal)
    assert jnp.allclose(normal_direction, expected, atol=1e-6)
