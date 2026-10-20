"""Regression checks for normals under per-object local scaling."""

from jax import Array
import jax.numpy as jnp
import numpy as np
import pytest

from renderer._meta_utils import typed_asarray, typed_grad, typed_jit
from renderer.geometry import normalise
from renderer.model import MergedModel, Model, ModelObject, merge_objects
from renderer.renderer import CameraParameters, LightParameters, Renderer
from renderer.shapes.cube import create_cube
from renderer.types import FaceIndices, Normals, Texture, UVCoordinates, Vertices


def _float_array(values: object) -> Array:
    return typed_asarray(values, dtype=jnp.float32)


@pytest.mark.parametrize(
    "local_scaling",
    ((2.0, 0.5, 3.0), (-2.0, 0.5, 3.0), (0.0, 1.0, 1.0)),
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
    # while inverse-transpose normals do not. Zero scale components count as
    # positive when choosing that sign.
    expected: Array = jnp.cross(transformed_tangents[0], transformed_tangents[1])
    scale_sign = jnp.prod(jnp.where(_float_array(local_scaling) < 0.0, -1.0, 1.0))
    expected = expected * scale_sign
    expected = normalise(expected)
    normal_direction = normalise(normal)
    assert jnp.allclose(normal_direction, expected, atol=1e-6)


@pytest.mark.parametrize(
    "local_scaling",
    ((2.0, 0.5, 3.0), (-2.0, 0.5, 3.0), (1.0, 1.0, 1.0)),
)
@pytest.mark.parametrize("unit_directions", (True, False))
def test_merge_objects_scaled_normal_gradient_matches_finite_differences(
    local_scaling: tuple[float, float, float],
    unit_directions: bool,
) -> None:
    verts: Vertices = _float_array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    norms: Normals = _float_array(((1.0, 2.0, 3.0), (-2.0, 1.0, 1.0), (0.5, -1.0, 2.0)))
    uvs: UVCoordinates = _float_array(((0.0, 0.0),) * 3)
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=jnp.int32)
    diffuse_map: Texture = _float_array((((1.0, 1.0, 1.0),),))
    model = Model.create(verts, norms, uvs, faces, diffuse_map)
    obj = ModelObject(model=model)
    weights: Array = _float_array(
        ((1.0, -0.5, 0.25), (-0.75, 0.5, 1.0), (0.3, 0.8, -0.4))
    )

    def loss(scale: Array) -> Array:
        merged = merge_objects((obj._replace(local_scaling=scale),))
        if unit_directions:
            values = jnp.stack(
                tuple(normalise(merged.norms[index]) for index in range(3))
            )
        else:
            values = merged.norms
        return jnp.sum(values * weights)

    def inverse_transpose_loss(scale: Array) -> Array:
        expected = jnp.stack(
            tuple(normalise(norms[index] / scale) for index in range(3))
        )
        return jnp.sum(expected * weights)

    scale = _float_array(local_scaling)
    gradient: Array = typed_jit(typed_grad(loss))(scale)
    epsilon = _float_array(1e-3)
    finite_difference = jnp.stack(
        tuple(
            (loss(scale.at[index].add(epsilon)) - loss(scale.at[index].add(-epsilon)))
            / (2.0 * epsilon)
            for index in range(3)
        )
    )

    assert jnp.linalg.norm(gradient) > 1e-2
    assert jnp.allclose(gradient, finite_difference, atol=1e-4, rtol=2e-3)
    if unit_directions:
        baseline_gradient: Array = typed_jit(typed_grad(inverse_transpose_loss))(scale)
        assert jnp.allclose(gradient, baseline_gradient, atol=1e-5, rtol=1e-4)


def test_merge_objects_normal_component_gradient_matches_finite_differences() -> None:
    verts: Vertices = _float_array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    source_normal = _float_array((0.0, 1.0, 1.0))
    norms: Normals = jnp.broadcast_to(source_normal, (3, 3))
    uvs: UVCoordinates = _float_array(((0.0, 0.0),) * 3)
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=jnp.int32)
    diffuse_map: Texture = _float_array((((1.0, 1.0, 1.0),),))
    model = Model.create(verts, norms, uvs, faces, diffuse_map)
    obj = ModelObject(model=model, local_scaling=_float_array((2.0, 0.5, 3.0)))
    weights: Array = _float_array(
        ((1.0, -0.5, 0.25), (-0.75, 0.5, 1.0), (0.3, 0.8, -0.4))
    )

    def loss(source_normals: Normals) -> Array:
        merged = merge_objects(
            (obj._replace(model=model._replace(norms=source_normals)),)
        )
        directions = jnp.stack(
            tuple(normalise(merged.norms[index]) for index in range(3))
        )
        return jnp.sum(directions * weights)

    def inverse_transpose_loss(source_normals: Normals) -> Array:
        expected = jnp.stack(
            tuple(
                normalise(source_normals[index] / obj.local_scaling)
                for index in range(3)
            )
        )
        return jnp.sum(expected * weights)

    gradient: Array = typed_jit(typed_grad(loss))(norms)
    baseline_gradient: Array = typed_jit(typed_grad(inverse_transpose_loss))(norms)
    epsilon = _float_array(1e-3)
    finite_difference = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    (
                        loss(norms.at[row, column].add(epsilon))
                        - loss(norms.at[row, column].add(-epsilon))
                    )
                    / (2.0 * epsilon)
                    for column in range(3)
                )
            )
            for row in range(3)
        )
    )

    assert jnp.abs(gradient[0, 0]) > 1e-2
    assert jnp.allclose(gradient, baseline_gradient, atol=1e-5, rtol=1e-4)
    assert jnp.allclose(gradient, finite_difference, atol=1e-4, rtol=2e-3)


@pytest.mark.parametrize(
    ("local_scaling", "source_normal", "other_normals"),
    (
        (
            (0.0, 2.0, 3.0),
            (0.0, 1.0, 1.0),
            ((1.0, 0.25, -0.5), (-0.5, 1.5, 0.75)),
        ),
        (
            (0.0, -2.0, 3.0),
            (0.0, 1.0, 1.0),
            ((1.0, 0.25, -0.5), (-0.5, 1.5, 0.75)),
        ),
        (
            (0.0, 0.0, -3.0),
            (0.0, 0.0, 1.0),
            ((1.0, 0.5, 0.25), (0.3, -1.0, 0.8)),
        ),
        (
            (0.0, 1e20, 1.0),
            (0.0, 1.0, 0.0),
            ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        (
            (1.0, 1e-12, 1e-12),
            (1.0, 0.0, 0.0),
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ),
        (
            (1.0, 1.0, 1e-24),
            (1.0, 0.0, 0.0),
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ),
    ),
)
def test_merge_objects_scaled_normal_direction_and_gradient_matches_limit(
    local_scaling: tuple[float, float, float],
    source_normal: tuple[float, float, float],
    other_normals: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> None:
    verts: Vertices = _float_array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    norms: Normals = _float_array((source_normal, *other_normals))
    uvs: UVCoordinates = _float_array(((0.0, 0.0),) * 3)
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=jnp.int32)
    diffuse_map: Texture = _float_array((((1.0, 1.0, 1.0),),))
    model = Model.create(verts, norms, uvs, faces, diffuse_map)
    obj = ModelObject(model=model)
    weights: Array = _float_array((0.31, -0.73, 1.17))

    def loss(scale: Array) -> Array:
        merged = merge_objects((obj._replace(local_scaling=scale),))
        target_direction = normalise(merged.norms[0])
        return jnp.sum(target_direction * weights)

    def inverse_transpose_direction(scale: Array) -> Array:
        safe_scale = jnp.where(scale == 0.0, 1.0, scale)
        source = _float_array(source_normal)
        if sum(value != 0.0 for value in source_normal) == 1:
            return normalise(source * jnp.sign(safe_scale))
        return normalise(source / safe_scale)

    def inverse_transpose_loss(scale: Array) -> Array:
        return jnp.sum(inverse_transpose_direction(scale) * weights)

    scale = _float_array(local_scaling)
    merged = merge_objects((obj._replace(local_scaling=scale),))
    actual_direction = normalise(merged.norms[0])
    expected_direction = inverse_transpose_direction(scale)
    gradient: Array = typed_jit(typed_grad(loss))(scale)
    baseline_gradient: Array = typed_jit(typed_grad(inverse_transpose_loss))(scale)
    epsilon = jnp.where(scale == 0.0, 1e-3, jnp.abs(scale) * 1e-3)
    finite_difference = jnp.stack(
        tuple(
            (
                loss(scale.at[index].add(epsilon[index]))
                - loss(scale.at[index].add(-epsilon[index]))
            )
            / (2.0 * epsilon[index])
            for index in range(3)
        )
    )

    assert jnp.all(jnp.isfinite(actual_direction))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.isfinite(finite_difference))
    assert jnp.allclose(actual_direction, expected_direction, atol=1e-6)
    assert jnp.allclose(gradient, baseline_gradient, atol=1e-5, rtol=1e-4)
    assert jnp.allclose(gradient, finite_difference, atol=1e-4, rtol=2e-3)
    if sum(value != 0.0 for value in source_normal) > 1 and local_scaling[1] != 0.0:
        assert jnp.linalg.norm(gradient[1:]) > 1e-2


def test_merge_objects_extreme_nonzero_scaling_keeps_inverse_transpose_direction() -> (
    None
):
    source_normal = (0.0, 1.0, 1.0)
    normals: Normals = _float_array((source_normal,) * 3)
    verts: Vertices = _float_array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    uvs: UVCoordinates = _float_array(((0.0, 0.0),) * 3)
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=jnp.int32)
    diffuse_map: Texture = _float_array((((1.0, 1.0, 1.0),),))
    model = Model.create(verts, normals, uvs, faces, diffuse_map)
    scale = (1e30, 1e-20, 1e-25)
    merged = merge_objects(
        (ModelObject(model=model, local_scaling=_float_array(scale)),)
    )

    normal64 = np.asarray(source_normal, dtype=np.float64)
    scale64 = np.asarray(scale, dtype=np.float64)
    expected64 = normal64 / scale64
    expected64 /= np.linalg.norm(expected64)
    actual_direction = normalise(merged.norms[0])
    expected_direction = _float_array(expected64)

    assert jnp.all(jnp.isfinite(merged.norms))
    assert jnp.all(jnp.isfinite(actual_direction))
    assert jnp.allclose(actual_direction, expected_direction, atol=1e-6)


def test_merge_objects_singular_normal_can_have_opposite_signed_limits() -> None:
    source_normal = _float_array((1.0, 0.0, 1.0))
    normals: Normals = jnp.broadcast_to(source_normal, (3, 3))
    verts: Vertices = _float_array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    uvs: UVCoordinates = _float_array(((0.0, 0.0),) * 3)
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=jnp.int32)
    diffuse_map: Texture = _float_array((((1.0, 1.0, 1.0),),))
    model = Model.create(verts, normals, uvs, faces, diffuse_map)
    obj = ModelObject(model=model)

    at_singular_scale = merge_objects(
        (obj._replace(local_scaling=_float_array((0.0, 0.0, 1.0))),)
    )
    positive_limit = merge_objects(
        (obj._replace(local_scaling=_float_array((1e-4, 1e-4, 1.0))),)
    )
    negative_limit = merge_objects(
        (obj._replace(local_scaling=_float_array((-1e-4, 1e-4, 1.0))),)
    )
    positive_direction = normalise(positive_limit.norms[0])
    negative_direction = normalise(negative_limit.norms[0])

    assert jnp.all(jnp.isfinite(at_singular_scale.norms))
    assert jnp.all(jnp.isfinite(positive_direction))
    assert jnp.all(jnp.isfinite(negative_direction))
    assert positive_direction[0] > 0.999
    assert negative_direction[0] < -0.999


def test_flattened_cube_keeps_finite_normals_and_renders_foreground() -> None:
    diffuse_map = _float_array((((0.2, 0.5, 0.8),),))
    specular_map = _float_array(((2.0,),))
    cube = create_cube(
        half_extents=_float_array((1.0, 1.0, 1.0)),
        texture_scaling=_float_array((1.0, 1.0)),
        diffuse_map=diffuse_map,
        specular_map=specular_map,
    )
    obj = ModelObject(model=cube, local_scaling=_float_array((0.0, 1.0, 1.0)))
    merged = merge_objects((obj,))

    assert jnp.all(jnp.isfinite(merged.norms))
    fully_collapsed = merge_objects(
        (obj._replace(local_scaling=_float_array((0.0, 0.0, 0.0))),)
    )
    assert jnp.all(jnp.isfinite(fully_collapsed.norms))

    camera = Renderer.create_camera_from_parameters(
        CameraParameters(
            viewWidth=24,
            viewHeight=24,
            position=(3.0, 0.0, 0.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        )
    )
    buffers = Renderer.render(
        model=merged,
        light=LightParameters(direction=_float_array((1.0, 0.0, 0.0))),
        camera=camera,
        buffers=Renderer.create_buffers(24, 24),
    )
    canvas: Array = buffers.targets[0]
    assert jnp.all(jnp.isfinite(canvas))
    assert jnp.any(canvas < 0.99)
