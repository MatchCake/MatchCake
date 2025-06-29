import numpy as np
import pennylane as qml
import pytest

from matchcake.utils import math

from ..configs import (ATOL_MATRIX_COMPARISON, ATOL_SHAPE_COMPARISON,
                       N_RANDOM_TESTS_PER_CASE, RTOL_MATRIX_COMPARISON,
                       RTOL_SHAPE_COMPARISON, TEST_SEED, set_seed)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "matrix,n,index,expected",
    [
        (
            np.array([[1, 2], [3, 4]]),
            4,
            1,
            np.array([[1, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 1]]),
        ),
        (
            np.array([[1, 2], [3, 4]]),
            8,
            2,
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 0, 0, 0, 0],
                    [0, 0, 3, 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            ),
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
            8,
            2,
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 3, 4, 0, 0],
                    [0, 0, 5, 6, 7, 8, 0, 0],
                    [0, 0, 9, 10, 11, 12, 0, 0],
                    [0, 0, 13, 14, 15, 16, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_eye_block_matrix_specific_cases(matrix, n, index, expected):
    result = math.eye_block_matrix(matrix, n, index)

    np.testing.assert_allclose(
        qml.math.shape(result),
        qml.math.shape(expected),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg=f"The shape of the result is not as expected. "
        f"Expected: {qml.math.shape(expected)}, got: {qml.math.shape(result)}.",
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg=f"The result is not as expected. Expected: {expected}, got: {result}.",
    )

    assert qml.math.get_interface(result) == qml.math.get_interface(matrix)
