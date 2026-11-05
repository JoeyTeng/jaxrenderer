"""Runtime behaviour of the core shader stages."""

from dataclasses import replace
from typing import NamedTuple, override

import jax
from jax import Array
from jaxtyping import Bool, Float
import numpy as np
import pytest

from renderer._meta_utils import typed_asarray, typed_grad, typed_jit
from renderer.shader import (
    Fragment,
    FragmentCollection,
    InterpolationInput,
    MixInput,
    MixerOutput,
    PassthroughShader,
    PrimitiveInput,
)
from renderer.shaders.gouraud import (
    GouraudExtraMixerOutput,
    GouraudShader,
)
from renderer.types import TRUE_ARRAY, PixelState


class _Varying(NamedTuple):
    colour: Array
    uv: Array


class _TestShader(PassthroughShader[tuple[Array, ...], _Varying]):
    pass


class _DoubleSidedShader(_TestShader):
    @staticmethod
    @override
    def primitive_chooser(
        inputs: PrimitiveInput[_Varying],
    ) -> tuple[
        Float[Array, "kept_primitives 4"],
        Bool[Array, "kept_primitives"],
        Float[Array, "kept_primitives 2"],
        Bool[Array, "kept_primitives"],
        _Varying,
        Float[Array, "kept_primitives 3"],
        Float[Array, "kept_primitives 3"],
    ]:
        return _TestShader.primitive_chooser(replace(inputs, double_sided=TRUE_ARRAY))


class _DestinationMixShader(_TestShader):
    @staticmethod
    @override
    def mix(inputs: MixInput[_Varying]) -> tuple[MixerOutput, _Varying]:
        return (
            MixerOutput(
                keep=inputs.keeps[0],
                zbuffer=inputs.destination.zbuffer + inputs.gl_FragDepth[0],
                write_depth=_array(False, dtype=np.bool_),
            ),
            _Varying(
                colour=(
                    inputs.destination.targets.colour + inputs.extra.targets.colour[0]
                ),
                uv=inputs.destination.targets.uv + inputs.extra.targets.uv[0],
            ),
        )


def _array(values: object, dtype: type[np.generic] = np.float32) -> Array:
    return typed_asarray(values, dtype=dtype)


def test_primitive_chooser_keeps_nearest_front_facing_candidate_and_structure() -> None:
    frag_coord = _array(
        [
            [0.0, 0.0, 0.05, 1.0],
            [0.0, 0.0, 0.1, 1.0],
            [0.0, 0.0, 0.7, 1.0],
            [0.0, 0.0, 0.2, 1.0],
        ]
    )
    point_coord = _array(np.arange(8, dtype=np.float32).reshape((4, 2)))
    values = _Varying(
        colour=_array(np.arange(12, dtype=np.float32).reshape((4, 3))),
        uv=_array(np.arange(8, dtype=np.float32).reshape((4, 2))),
    )
    barycentric_screen = _array(np.arange(12, dtype=np.float32).reshape((4, 3)))
    barycentric_clip = barycentric_screen + 10.0

    chosen = _TestShader.primitive_chooser(
        PrimitiveInput(
            gl_FragCoord=frag_coord,
            gl_FrontFacing=_array([False, True, True, True], dtype=np.bool_),
            gl_PointCoord=point_coord,
            keeps=_array([True, False, True, True], dtype=np.bool_),
            values=values,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
        )
    )

    assert chosen[0].shape == (1, 4)
    assert chosen[3].shape == (1,)
    assert isinstance(chosen[4], _Varying)
    np.testing.assert_array_equal(chosen[0], frag_coord[3:4])
    np.testing.assert_array_equal(chosen[1], _array([True], dtype=np.bool_))
    np.testing.assert_array_equal(chosen[2], point_coord[3:4])
    np.testing.assert_array_equal(chosen[3], _array([True], dtype=np.bool_))
    np.testing.assert_array_equal(chosen[4].colour, values.colour[3:4])
    np.testing.assert_array_equal(chosen[4].uv, values.uv[3:4])
    np.testing.assert_array_equal(chosen[5], barycentric_screen[3:4])
    np.testing.assert_array_equal(chosen[6], barycentric_clip[3:4])

    for double_sided in (
        _array(True, dtype=np.bool_),
        _array([True, False, False, False], dtype=np.bool_),
    ):
        chosen = _TestShader.primitive_chooser(
            PrimitiveInput(
                gl_FragCoord=frag_coord,
                gl_FrontFacing=_array([False, True, True, True], dtype=np.bool_),
                gl_PointCoord=point_coord,
                keeps=_array([True, False, True, True], dtype=np.bool_),
                values=values,
                barycentric_screen=barycentric_screen,
                barycentric_clip=barycentric_clip,
                double_sided=double_sided,
            )
        )

        assert chosen[0].shape == (1, 4)
        assert isinstance(chosen[4], _Varying)
        np.testing.assert_array_equal(chosen[0], frag_coord[0:1])
        np.testing.assert_array_equal(chosen[1], _array([False], dtype=np.bool_))
        np.testing.assert_array_equal(chosen[2], point_coord[0:1])
        np.testing.assert_array_equal(chosen[3], _array([True], dtype=np.bool_))
        np.testing.assert_array_equal(chosen[4].colour, values.colour[0:1])
        np.testing.assert_array_equal(chosen[4].uv, values.uv[0:1])
        np.testing.assert_array_equal(chosen[5], barycentric_screen[0:1])
        np.testing.assert_array_equal(chosen[6], barycentric_clip[0:1])


def test_all_collection_retains_batch_slots_and_returns_eligibility_mask() -> None:
    frag_coord = _array(
        [
            [0.0, 0.0, 0.05, 1.0],
            [0.0, 0.0, 0.1, 1.0],
            [0.0, 0.0, 0.7, 1.0],
            [0.0, 0.0, 0.2, 1.0],
        ]
    )
    point_coord = _array(np.arange(8, dtype=np.float32).reshape((4, 2)))
    values = _Varying(
        colour=_array(np.arange(12, dtype=np.float32).reshape((4, 3))),
        uv=_array(np.arange(8, dtype=np.float32).reshape((4, 2))),
    )
    barycentric_screen = _array(np.arange(12, dtype=np.float32).reshape((4, 3)))
    barycentric_clip = barycentric_screen + 10.0

    chosen = _TestShader.primitive_chooser(
        PrimitiveInput(
            gl_FragCoord=frag_coord,
            gl_FrontFacing=_array([False, True, True, True], dtype=np.bool_),
            gl_PointCoord=point_coord,
            keeps=_array([True, False, True, True], dtype=np.bool_),
            values=values,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
            collection=FragmentCollection.ALL,
        )
    )

    assert chosen[0].shape == (4, 4)
    assert chosen[3].shape == (4,)
    np.testing.assert_array_equal(chosen[0], frag_coord)
    np.testing.assert_array_equal(chosen[1], [False, True, True, True])
    np.testing.assert_array_equal(chosen[2], point_coord)
    np.testing.assert_array_equal(chosen[3], [False, False, True, True])
    np.testing.assert_array_equal(chosen[4].colour, values.colour)
    np.testing.assert_array_equal(chosen[4].uv, values.uv)
    np.testing.assert_array_equal(chosen[5], barycentric_screen)
    np.testing.assert_array_equal(chosen[6], barycentric_clip)


def test_interpolate_preserves_namedtuple_and_uses_smooth_weights() -> None:
    values = _Varying(
        colour=_array([[2.0, 4.0], [8.0, 16.0], [20.0, 32.0]]),
        uv=_array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]),
    )
    barycentric_screen = _array([0.2, 0.3, 0.5])
    barycentric_clip = _array([0.6, 0.1, 0.3])

    result = _TestShader.interpolate(
        InterpolationInput(
            values=values,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
        )
    )

    assert isinstance(result, _Varying)
    np.testing.assert_allclose(result.colour, [8.0, 13.6])
    np.testing.assert_allclose(result.uv, [3.0, 5.0])


def test_interpolation_input_supports_jit_vmap_and_autodiff() -> None:
    inputs = InterpolationInput(
        values=_Varying(
            colour=_array([[[1.0], [2.0], [4.0]], [[2.0], [4.0], [8.0]]]),
            uv=_array([[[0.0], [2.0], [4.0]], [[1.0], [3.0], [5.0]]]),
        ),
        barycentric_screen=_array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]]),
        barycentric_clip=_array([[0.6, 0.1, 0.3], [0.2, 0.3, 0.5]]),
    )
    interpolate_batch = typed_jit(jax.vmap(_TestShader.interpolate))
    result = interpolate_batch(inputs)

    assert isinstance(result, _Varying)
    assert result.colour.shape == (2, 1)

    def total_colour(scale: Array) -> Array:
        scaled_inputs = replace(
            inputs,
            values=inputs.values._replace(colour=inputs.values.colour * scale),
        )
        return interpolate_batch(scaled_inputs).colour.sum()

    np.testing.assert_allclose(
        typed_grad(total_colour)(_array(1.0)), result.colour.sum()
    )


def test_primitive_chooser_replace_forwards_to_inherited_stage() -> None:
    inputs = PrimitiveInput(
        gl_FragCoord=_array([[0.0, 0.0, 0.1, 1.0], [0.0, 0.0, 0.2, 1.0]]),
        gl_FrontFacing=_array([False, True], dtype=np.bool_),
        gl_PointCoord=_array(np.zeros((2, 2), dtype=np.float32)),
        keeps=_array([True, True], dtype=np.bool_),
        values=_Varying(
            colour=_array([[1.0], [2.0]]),
            uv=_array([[3.0], [4.0]]),
        ),
        barycentric_screen=_array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        barycentric_clip=_array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        collection=FragmentCollection.ALL,
    )

    chosen = _DoubleSidedShader.primitive_chooser(inputs)

    np.testing.assert_array_equal(chosen[0], inputs.gl_FragCoord)
    np.testing.assert_array_equal(chosen[1], inputs.gl_FrontFacing)
    np.testing.assert_array_equal(chosen[3], [True, True])


def test_fragment_defaults_to_opaque_alpha() -> None:
    fragment = Fragment(targets=_Varying(colour=_array([1.0]), uv=_array([2.0])))

    np.testing.assert_array_equal(fragment.alpha, 1.0)


def test_mix_chooses_nearest_kept_candidate_and_empty_uses_destination() -> None:
    depths = _array([0.9, 0.1, 0.4])
    keeps = _array([True, False, True], dtype=np.bool_)
    extra = Fragment(
        targets=_Varying(
            colour=_array([[1.0], [2.0], [3.0]]),
            uv=_array([[10.0], [20.0], [30.0]]),
        )
    )
    destination = PixelState(
        zbuffer=_array(0.8),
        targets=_Varying(colour=_array([8.0]), uv=_array([80.0])),
    )

    selected, selected_targets = _TestShader.mix(
        MixInput(
            gl_FragDepth=depths,
            keeps=keeps,
            extra=extra,
            destination=destination,
        )
    )

    assert selected.keep
    assert selected.write_depth
    np.testing.assert_allclose(selected.zbuffer, 0.4)
    assert isinstance(selected_targets, _Varying)
    np.testing.assert_array_equal(selected_targets.colour, extra.targets.colour[2])
    np.testing.assert_array_equal(selected_targets.uv, extra.targets.uv[2])

    empty, empty_targets = _TestShader.mix(
        MixInput(
            gl_FragDepth=depths,
            keeps=_array([False, False, False], dtype=np.bool_),
            extra=extra,
            destination=destination,
        )
    )

    assert not empty.keep
    assert not empty.write_depth
    assert isinstance(empty_targets, _Varying)

    zero_candidates, zero_targets = _TestShader.mix(
        MixInput(
            gl_FragDepth=_array([], dtype=np.float32),
            keeps=_array([], dtype=np.bool_),
            extra=Fragment(
                targets=_Varying(
                    colour=_array(np.zeros((0, 1), dtype=np.float32)),
                    uv=_array(np.zeros((0, 1), dtype=np.float32)),
                )
            ),
            destination=destination,
        ),
    )
    assert not zero_candidates.keep
    assert not zero_candidates.write_depth
    np.testing.assert_array_equal(zero_candidates.zbuffer, destination.zbuffer)
    assert isinstance(zero_targets, _Varying)
    np.testing.assert_array_equal(zero_targets.colour, destination.targets.colour)
    np.testing.assert_array_equal(zero_targets.uv, destination.targets.uv)


def test_gouraud_inherits_typed_mix_output() -> None:
    fragment = Fragment(
        targets=GouraudExtraMixerOutput(canvas=_array([[0.2, 0.4, 0.8]]))
    )
    destination = PixelState(
        zbuffer=_array(1.0),
        targets=GouraudExtraMixerOutput(canvas=_array([0.0, 0.0, 0.0])),
    )

    output, targets = GouraudShader.mix(
        MixInput(
            gl_FragDepth=_array([0.25]),
            keeps=_array([True], dtype=np.bool_),
            extra=fragment,
            destination=destination,
        )
    )

    assert output.keep
    assert output.write_depth
    assert isinstance(targets, GouraudExtraMixerOutput)
    np.testing.assert_allclose(targets.canvas, [0.2, 0.4, 0.8])


def test_custom_mix_reads_destination_and_can_suppress_depth_write() -> None:
    extra = Fragment(targets=_Varying(colour=_array([[1.0, 2.0]]), uv=_array([[3.0]])))
    destination = PixelState(
        zbuffer=_array(0.6),
        targets=_Varying(colour=_array([4.0, 5.0]), uv=_array([6.0])),
    )

    output, targets = _DestinationMixShader.mix(
        MixInput(
            gl_FragDepth=_array([0.2]),
            keeps=_array([True], dtype=np.bool_),
            extra=extra,
            destination=destination,
        )
    )

    assert output.keep
    assert not output.write_depth
    np.testing.assert_allclose(output.zbuffer, 0.8)
    np.testing.assert_allclose(targets.colour, [5.0, 7.0])
    np.testing.assert_allclose(targets.uv, [9.0])


def test_primitive_chooser_rejects_non_boolean_facing_flags() -> None:
    frag_coord = _array(np.zeros((2, 4), dtype=np.float32))
    point_coord = _array(np.zeros((2, 2), dtype=np.float32))
    values = _Varying(
        colour=_array(np.zeros((2, 3), dtype=np.float32)),
        uv=_array(np.zeros((2, 2), dtype=np.float32)),
    )
    barycentric = _array(np.zeros((2, 3), dtype=np.float32))

    with pytest.raises((AssertionError, TypeError)):
        _ = _TestShader.primitive_chooser(
            PrimitiveInput(
                gl_FragCoord=frag_coord,
                gl_FrontFacing=_array([1, 0], dtype=np.int32),
                gl_PointCoord=point_coord,
                keeps=_array([True, True], dtype=np.bool_),
                values=values,
                barycentric_screen=barycentric,
                barycentric_clip=barycentric,
            )
        )
