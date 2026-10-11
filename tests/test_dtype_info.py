from typing import cast

import jax.numpy as jnp
import pytest

from renderer._backport import JaxFloating, JaxInteger, Type
from renderer.types import DtypeInfo


def _assert_float_info(
    dtype: Type[JaxFloating], bits: int, minimum: float, maximum: float
) -> None:
    info = DtypeInfo[JaxFloating].create(dtype)

    assert info.dtype is dtype
    assert info.min == minimum
    assert info.max == maximum
    assert info.bits == bits


def test_dtype_info_reports_float32_limits() -> None:
    _assert_float_info(jnp.float32, 32, -3.4028234663852886e38, 3.4028234663852886e38)


def test_dtype_info_reports_float64_limits() -> None:
    _assert_float_info(jnp.float64, 64, -1.7976931348623157e308, 1.7976931348623157e308)


def test_dtype_info_reports_bfloat16_limits() -> None:
    _assert_float_info(jnp.bfloat16, 16, -3.3895313892515355e38, 3.3895313892515355e38)


def _assert_integer_info(
    dtype: Type[JaxInteger], bits: int, minimum: int, maximum: int
) -> None:
    info = DtypeInfo[JaxInteger].create(dtype)

    assert info.dtype is dtype
    assert info.min == minimum
    assert info.max == maximum
    assert info.bits == bits


def test_dtype_info_reports_int32_limits() -> None:
    _assert_integer_info(jnp.int32, 32, -(2**31), 2**31 - 1)


def test_dtype_info_reports_uint32_limits() -> None:
    _assert_integer_info(jnp.uint32, 32, 0, 2**32 - 1)


def test_dtype_info_accepts_python_int() -> None:
    info = DtypeInfo[int].create(int)

    assert info.dtype is int
    assert info.min == jnp.iinfo(int).min
    assert info.max == jnp.iinfo(int).max
    assert info.bits == jnp.iinfo(int).bits


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.complex64])
def test_dtype_info_rejects_unsupported_types(dtype: object) -> None:
    with pytest.raises(ValueError, match="Unexpected dtype"):
        _ = DtypeInfo[int].create(cast(Type[int], dtype))
