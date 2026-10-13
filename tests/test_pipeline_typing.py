from typing import NamedTuple, override

from jax import Array
import jax.numpy as jnp
import numpy as np

from renderer import Buffers, Camera, render
from renderer.geometry import Viewport
from renderer.shader import ID, PerVertex, Shader
from renderer.types import Canvas, ZBuffer


class _Inputs(NamedTuple):
    positions: Array
    colours: Array
    scalar_values: Array


class _Varyings(NamedTuple):
    colour: Array
    scalar: Array


class _MultiTargetShader(Shader[_Inputs, _Varyings, _Varyings]):
    @staticmethod
    @override
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: _Inputs,
    ) -> tuple[PerVertex, _Varyings]:
        del gl_InstanceID, camera
        return (
            PerVertex(gl_Position=extra.positions[gl_VertexID]),
            _Varyings(
                colour=extra.colours[gl_VertexID],
                scalar=extra.scalar_values[gl_VertexID],
            ),
        )


def _array(values: object) -> Array:
    return jnp.asarray(values, dtype=jnp.float32)  # pyright: ignore[reportUnknownMemberType]


def test_render_updates_multiple_targets_and_preserves_background() -> None:
    width = 10
    height = 10
    viewport: Viewport = Camera.viewport_matrix(
        lowerbound=jnp.asarray((0, 0), dtype=int),  # pyright: ignore[reportUnknownMemberType]
        dimension=jnp.asarray((width, height), dtype=int),  # pyright: ignore[reportUnknownMemberType]
        depth=1,
    )
    camera = Camera.create(
        view=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        projection=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        viewport=viewport,
    )

    background_colour = (0.1, 0.2, 0.3)
    background_scalar = 7.0
    background_depth = 1.0
    buffers: Buffers[tuple[Canvas, ZBuffer]] = Buffers(
        zbuffer=jnp.full((width, height), background_depth),  # pyright: ignore[reportUnknownMemberType]
        targets=(
            jnp.full(  # pyright: ignore[reportUnknownMemberType]
                (width, height, 3), _array(background_colour)
            ),
            jnp.full((width, height), background_scalar),  # pyright: ignore[reportUnknownMemberType]
        ),
    )
    extra = _Inputs(
        positions=_array(
            (
                (-0.8, -0.8, 0.0, 1.0),
                (0.8, -0.8, 0.0, 1.0),
                (-0.8, 0.8, 0.0, 1.0),
            )
        ),
        colours=_array(((0.8, 0.4, 0.2),) * 3),
        scalar_values=_array((0.6,) * 3),
    )
    face_indices = jnp.asarray(((0, 1, 2),), dtype=int)  # pyright: ignore[reportUnknownMemberType]

    result = render(
        camera=camera,
        shader=_MultiTargetShader,
        buffers=buffers,
        face_indices=face_indices,
        extra=extra,
        loop_unroll=2,
    )

    assert isinstance(result.targets, tuple)
    assert len(result.targets) == 2
    colour_target, scalar_target = result.targets
    interior = (3, 3)
    outside = (9, 9)

    assert np.isfinite(np.asarray(colour_target)).all()
    assert np.isfinite(np.asarray(scalar_target)).all()
    np.testing.assert_allclose(colour_target[interior], (0.8, 0.4, 0.2))
    np.testing.assert_allclose(scalar_target[interior], 0.6)
    np.testing.assert_allclose(colour_target[outside], background_colour)
    np.testing.assert_allclose(scalar_target[outside], background_scalar)
    np.testing.assert_allclose(result.zbuffer[interior], 0.5)
    np.testing.assert_allclose(result.zbuffer[outside], background_depth)
