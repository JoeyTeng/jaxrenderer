from jax import Array
import jax.numpy as jnp
import numpy as np
import pytest

from renderer import replace_dict
from renderer.utils import transpose_for_display


def test_replace_dict_gives_new_values_precedence_without_mutating_inputs() -> None:
    base = {"shared": "base", "base-only": "base"}
    new = {"shared": "new", "new-only": "new"}

    result = replace_dict(base, new)

    assert result == {
        "shared": "new",
        "base-only": "base",
        "new-only": "new",
    }
    assert result is not base
    assert result is not new
    assert base == {"shared": "base", "base-only": "base"}
    assert new == {"shared": "new", "new-only": "new"}


class TestTransposeForDisplay:
    @pytest.mark.parametrize(
        ("matrix", "flip_vertical", "expected"),
        [
            (
                np.asarray(
                    [
                        [0.1, 1.2, 2.3, 3.4, 4.5],
                        [5.6, 6.7, 7.8, 8.9, 9.1],
                        [10.2, 11.3, 12.4, 13.5, 14.6],
                        [15.7, 16.8, 17.9, 18.1, 19.2],
                    ],
                    dtype=np.float32,
                ),
                False,
                [
                    [0.1, 5.6, 10.2, 15.7],
                    [1.2, 6.7, 11.3, 16.8],
                    [2.3, 7.8, 12.4, 17.9],
                    [3.4, 8.9, 13.5, 18.1],
                    [4.5, 9.1, 14.6, 19.2],
                ],
            ),
            (
                np.asarray(
                    [[1.25, 2.5, 3.75], [4.5, 5.25, 6.125]],
                    dtype=np.float32,
                ),
                True,
                [[3.75, 6.125], [2.5, 5.25], [1.25, 4.5]],
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
        input_array = jnp.asarray(matrix)
        expected_array = np.asarray(expected, dtype=input_array.dtype)

        actual = transpose_for_display(input_array, flip_vertical=flip_vertical)

        assert isinstance(actual, Array)
        np.testing.assert_array_equal(actual, expected_array)
        assert actual.dtype == input_array.dtype
