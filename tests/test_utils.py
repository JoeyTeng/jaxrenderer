from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from jaxtyping import Array, Shaped, jaxtyped

from renderer.utils import transpose_for_display

PRNG_KEYS: list[random.KeyArray] = [random.key(20230701)]


class TestTransposeForDisplay:

    @pytest.mark.parametrize("PRNG_KEY", PRNG_KEYS)
    @pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 1), (20, 31, 3),
                                       (11, 11, 4)])
    @pytest.mark.parametrize("flip_vertical", [True, False])
    @jaxtyped
    def test_transposed_shape_must_be_flipped_along_first_two_axis(
        self,
        PRNG_KEY: random.KeyArray,
        shape: tuple[int, ...],
        flip_vertical: bool,
    ):
        matrix: Shaped[Array, "fst snd *channel"]
        matrix = random.uniform(PRNG_KEY, shape)
        transposed: Shaped[Array, "snd fst *channel"]
        transposed = transpose_for_display(matrix, flip_vertical=flip_vertical)

        assert (np.array(matrix.shape) == np.array(shape)).all(), (
            "Matrix shape must not be changed")
        assert (np.array(transposed.shape) == np.array([
            shape[1], shape[0], *shape[2:]
        ])).all(), ("Transposed shape must be flipped along first two axiss")
        assert isinstance(matrix, Shaped[Array, "fst snd *channel"])
        assert isinstance(transposed, Shaped[Array, "snd fst *channel"])

    @pytest.mark.parametrize("PRNG_KEY", PRNG_KEYS)
    @pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 1), (20, 31, 3),
                                       (11, 11, 4)])
    @pytest.mark.parametrize("flip_vertical", [True, False])
    @jaxtyped
    def test_transposed_unique_values_and_count_must_be_the_same(
        self,
        PRNG_KEY: random.KeyArray,
        shape: tuple[int, ...],
        flip_vertical: bool,
    ):
        matrix: Shaped[Array, "fst snd *channel"]
        matrix = random.uniform(PRNG_KEY, shape)
        transposed: Shaped[Array, "snd fst *channel"]
        transposed = transpose_for_display(matrix, flip_vertical=flip_vertical)

        # Unique values: Transpose of the matrix should not change the values
        m, m_cnt = jnp.unique(matrix, return_counts=True)
        t, t_cnt = jnp.unique(transposed, return_counts=True)

        assert (m == t).all(), ("Unique values must be the same")
        assert (m_cnt == t_cnt).all(), ("Unique values count must be the same")

    @pytest.mark.parametrize("PRNG_KEY", PRNG_KEYS)
    @pytest.mark.parametrize("shape", [(5, 3), (20, 31, 3), (11, 11, 4)])
    @jaxtyped
    def test_flip_vertical(
        self,
        PRNG_KEY: random.KeyArray,
        shape: tuple[int, ...],
    ):
        # transpose with flipping
        tf_f = partial(transpose_for_display, flip_vertical=True)
        # transpose without flipping
        t_f = partial(transpose_for_display, flip_vertical=False)

        matrix: Shaped[Array, "fst snd *channel"]
        matrix = random.uniform(PRNG_KEY, shape)
        tf: Shaped[Array, "snd fst *channel"] = tf_f(matrix)
        t: Shaped[Array, "snd fst *channel"] = t_f(matrix)

        assert (t != tf).any(), ("flipped vertical will change the matrix")

        ttfttf = t_f(tf_f(t_f(tf_f(matrix))))
        assert (ttfttf == matrix).all(), (
            "flip twice, transpose 4 times should be identity")

        tt = t_f(t_f(matrix))
        assert (tt == matrix).all(), ("transpose twice should be identity")

        tftftftf = tf_f(tf_f(tf_f(tf_f(matrix))))
        assert (tftftftf == matrix).all(), (
            "transpose and flip 4 times should be identity")

        tftf = tf_f(tf_f(matrix))
        assert (tftf != matrix).any(), (
            "transpose and flip twice should not be identity")
