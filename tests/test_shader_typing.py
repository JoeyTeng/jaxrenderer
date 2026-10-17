"""Runtime behaviour of the core shader stages."""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
import numpy as np
import pytest

from renderer._meta_utils import typed_asarray
from renderer.shader import Shader


class _Varying(NamedTuple):
    colour: Array
    uv: Array


class _TestShader(Shader[tuple[Array, ...], _Varying, _Varying]):
    pass


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
        gl_FragCoord=frag_coord,
        gl_FrontFacing=_array([False, True, True, True], dtype=np.bool_),
        gl_PointCoord=point_coord,
        keeps=_array([True, False, True, True], dtype=np.bool_),
        values=values,
        barycentric_screen=barycentric_screen,
        barycentric_clip=barycentric_clip,
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


def test_interpolate_preserves_namedtuple_and_uses_smooth_weights() -> None:
    values = _Varying(
        colour=_array([[2.0, 4.0], [8.0, 16.0], [20.0, 32.0]]),
        uv=_array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]),
    )
    barycentric_screen = _array([0.2, 0.3, 0.5])
    barycentric_clip = _array([0.6, 0.1, 0.3])

    result = _TestShader.interpolate(values, barycentric_screen, barycentric_clip)

    assert isinstance(result, _Varying)
    np.testing.assert_allclose(result.colour, [8.0, 13.6])
    np.testing.assert_allclose(result.uv, [3.0, 5.0])


def test_mix_chooses_minimum_kept_depth_and_empty_set_keeps_infinite_depth() -> None:
    depths = _array([0.9, 0.1, 0.4])
    keeps = _array([True, False, True], dtype=np.bool_)
    extra = _Varying(
        colour=_array([[1.0], [2.0], [3.0]]),
        uv=_array([[10.0], [20.0], [30.0]]),
    )

    selected, selected_extra = _TestShader.mix(depths, keeps, extra)

    assert selected.keep
    np.testing.assert_allclose(selected.zbuffer, 0.4)
    assert isinstance(selected_extra, _Varying)
    np.testing.assert_array_equal(selected_extra.colour, extra.colour[2])
    np.testing.assert_array_equal(selected_extra.uv, extra.uv[2])

    empty, _ = _TestShader.mix(
        depths, _array([False, False, False], dtype=np.bool_), extra
    )

    assert not empty.keep
    assert jnp.isinf(empty.zbuffer)


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
            gl_FragCoord=frag_coord,
            gl_FrontFacing=_array([1, 0], dtype=np.int32),
            gl_PointCoord=point_coord,
            keeps=_array([True, True], dtype=np.bool_),
            values=values,
            barycentric_screen=barycentric,
            barycentric_clip=barycentric,
        )
