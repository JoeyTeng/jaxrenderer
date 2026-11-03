from typing import NamedTuple, override

from jax import Array
import jax.numpy as jnp
from jaxtyping import Bool, Float
import numpy as np

from renderer import Buffers, Camera, render
from renderer._meta_utils import typed_asarray, typed_full
from renderer.geometry import Viewport
from renderer.shader import (
    ID,
    Fragment,
    FragmentCollection,
    MixerOutput,
    PerFragment,
    PerVertex,
    Shader,
)
from renderer.types import (
    FALSE_ARRAY,
    TRUE_ARRAY,
    BoolV,
    Canvas,
    PixelState,
    Vec2f,
    Vec4f,
    ZBuffer,
)


class _Inputs(NamedTuple):
    positions: Array
    colours: Array
    scalar_values: Array


class _Varyings(NamedTuple):
    colour: Array
    scalar: Array


class _Targets(NamedTuple):
    colour: Canvas
    scalar: ZBuffer


class _MultiTargetShader(Shader[_Inputs, _Varyings, _Targets]):
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

    @staticmethod
    @override
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: _Varyings,
        extra: _Inputs,
    ) -> tuple[PerFragment, Fragment[_Targets]]:
        del gl_FragCoord, gl_FrontFacing, gl_PointCoord, extra
        return (
            PerFragment(use_default_depth=TRUE_ARRAY),
            Fragment(targets=_Targets(colour=varying.colour, scalar=varying.scalar)),
        )


class _TargetOnlyShader(_MultiTargetShader):
    @staticmethod
    @override
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: Fragment[_Targets],
        *,
        destination: PixelState[_Targets],
    ) -> tuple[MixerOutput, _Targets]:
        output, targets = _MultiTargetShader.mix(
            gl_FragDepth, keeps, extra, destination=destination
        )
        return (
            output._replace(write_depth=FALSE_ARRAY),
            _Targets(
                colour=targets.colour + destination.targets.colour,
                scalar=targets.scalar + destination.targets.scalar,
            ),
        )


class _DepthOnlyShader(_MultiTargetShader):
    @staticmethod
    @override
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: Fragment[_Targets],
        *,
        destination: PixelState[_Targets],
    ) -> tuple[MixerOutput, _Targets]:
        output, _targets = _MultiTargetShader.mix(
            gl_FragDepth, keeps, extra, destination=destination
        )
        return (
            output._replace(keep=FALSE_ARRAY),
            destination.targets,
        )


class _AllCandidatesShader(_MultiTargetShader):
    fragment_collection = FragmentCollection.ALL

    @staticmethod
    @override
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: _Varyings,
        extra: _Inputs,
    ) -> tuple[PerFragment, Fragment[_Targets]]:
        del gl_FrontFacing, gl_PointCoord, extra
        return (
            PerFragment(
                keeps=gl_FragCoord[2] >= 0.5,
                use_default_depth=TRUE_ARRAY,
            ),
            Fragment(targets=_Targets(colour=varying.colour, scalar=varying.scalar)),
        )


def _array(values: object) -> Array:
    return typed_asarray(values, dtype=jnp.float32)


def test_render_updates_multiple_targets_and_preserves_background() -> None:
    width = 10
    height = 10
    viewport: Viewport = Camera.viewport_matrix(
        lowerbound=typed_asarray((0, 0), dtype=int),
        dimension=typed_asarray((width, height), dtype=int),
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

    def initial_buffers() -> Buffers[_Targets]:
        return Buffers(
            zbuffer=typed_full((width, height), background_depth),
            targets=_Targets(
                colour=typed_full((width, height, 3), _array(background_colour)),
                scalar=typed_full((width, height), background_scalar),
            ),
        )

    buffers = initial_buffers()
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
    face_indices = typed_asarray(((0, 1, 2),), dtype=int)

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

    target_only = render(
        camera=camera,
        shader=_TargetOnlyShader,
        buffers=initial_buffers(),
        face_indices=face_indices,
        extra=extra,
        loop_unroll=2,
    )
    np.testing.assert_allclose(target_only.zbuffer[interior], background_depth)
    np.testing.assert_allclose(target_only.targets.colour[interior], (0.9, 0.6, 0.5))
    np.testing.assert_allclose(target_only.targets.scalar[interior], 7.6)

    depth_only = render(
        camera=camera,
        shader=_DepthOnlyShader,
        buffers=initial_buffers(),
        face_indices=face_indices,
        extra=extra,
        loop_unroll=2,
    )
    np.testing.assert_allclose(depth_only.zbuffer[interior], 0.5)
    np.testing.assert_allclose(depth_only.targets.colour[interior], background_colour)
    np.testing.assert_allclose(depth_only.targets.scalar[interior], background_scalar)


def test_all_collection_falls_back_after_nearer_fragment_discards() -> None:
    width = 10
    height = 10
    viewport: Viewport = Camera.viewport_matrix(
        lowerbound=typed_asarray((0, 0), dtype=int),
        dimension=typed_asarray((width, height), dtype=int),
        depth=1,
    )
    camera = Camera.create(
        view=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        projection=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        viewport=viewport,
    )
    near_colour = (0.9, 0.1, 0.1)
    far_colour = (0.1, 0.8, 0.2)
    extra = _Inputs(
        positions=_array(
            (
                (-0.8, -0.8, -0.5, 1.0),
                (0.8, -0.8, -0.5, 1.0),
                (-0.8, 0.8, -0.5, 1.0),
                (-0.8, -0.8, 0.5, 1.0),
                (0.8, -0.8, 0.5, 1.0),
                (-0.8, 0.8, 0.5, 1.0),
            )
        ),
        colours=_array((near_colour,) * 3 + (far_colour,) * 3),
        scalar_values=_array((1.0,) * 3 + (2.0,) * 3),
    )
    buffers = Buffers(
        zbuffer=typed_full((width, height), 1.0),
        targets=_Targets(
            colour=typed_full((width, height, 3), _array((0.0, 0.0, 0.0))),
            scalar=typed_full((width, height), 0.0),
        ),
    )

    result = render(
        camera=camera,
        shader=_AllCandidatesShader,
        buffers=buffers,
        face_indices=typed_asarray(((0, 1, 2), (3, 4, 5)), dtype=int),
        extra=extra,
        loop_unroll=2,
    )

    np.testing.assert_allclose(result.targets.colour[3, 3], far_colour)
    np.testing.assert_allclose(result.targets.scalar[3, 3], 2.0)
    np.testing.assert_allclose(result.zbuffer[3, 3], 0.75)
