import jax.numpy as jnp
import numpy as np
import pytest

from renderer.utils import transpose_for_display


class TestTransposeForDisplay:
    @pytest.mark.parametrize(
        ("matrix", "flip_vertical", "expected"),
        [
            (
                [[1, 2, 3], [4, 5, 6]],
                False,
                [[1, 4], [2, 5], [3, 6]],
            ),
            (
                [[1, 2, 3], [4, 5, 6]],
                True,
                [[3, 6], [2, 5], [1, 4]],
            ),
            ([[7, 8, 9]], False, [[7], [8], [9]]),
            ([[7, 8, 9]], True, [[9], [8], [7]]),
            ([[1], [2], [3]], False, [[1, 2, 3]]),
            ([[1], [2], [3]], True, [[1, 2, 3]]),
            (
                [
                    [[1, 11, 21], [2, 12, 22], [3, 13, 23]],
                    [[4, 14, 24], [5, 15, 25], [6, 16, 26]],
                ],
                True,
                [
                    [[3, 13, 23], [6, 16, 26]],
                    [[2, 12, 22], [5, 15, 25]],
                    [[1, 11, 21], [4, 14, 24]],
                ],
            ),
            (
                [
                    [[1, 11, 21], [2, 12, 22], [3, 13, 23]],
                    [[4, 14, 24], [5, 15, 25], [6, 16, 26]],
                ],
                False,
                [
                    [[1, 11, 21], [4, 14, 24]],
                    [[2, 12, 22], [5, 15, 25]],
                    [[3, 13, 23], [6, 16, 26]],
                ],
            ),
            (
                [
                    [[1, 11, 21, 31], [2, 12, 22, 32], [3, 13, 23, 33]],
                    [[4, 14, 24, 34], [5, 15, 25, 35], [6, 16, 26, 36]],
                ],
                False,
                [
                    [[1, 11, 21, 31], [4, 14, 24, 34]],
                    [[2, 12, 22, 32], [5, 15, 25, 35]],
                    [[3, 13, 23, 33], [6, 16, 26, 36]],
                ],
            ),
            (
                [
                    [[1, 11, 21, 31], [2, 12, 22, 32], [3, 13, 23, 33]],
                    [[4, 14, 24, 34], [5, 15, 25, 35], [6, 16, 26, 36]],
                ],
                True,
                [
                    [[3, 13, 23, 33], [6, 16, 26, 36]],
                    [[2, 12, 22, 32], [5, 15, 25, 35]],
                    [[1, 11, 21, 31], [4, 14, 24, 34]],
                ],
            ),
        ],
    )
    def test_transpose_and_optional_vertical_flip_match_expected_values(
        self,
        matrix: list,
        flip_vertical: bool,
        expected: list,
    ):
        input_array = jnp.asarray(matrix, dtype=jnp.int16)
        expected_array = np.asarray(expected, dtype=np.int16)

        actual = transpose_for_display(input_array, flip_vertical=flip_vertical)

        np.testing.assert_array_equal(actual, expected_array)
        assert actual.dtype == input_array.dtype
