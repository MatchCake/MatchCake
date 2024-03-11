import numpy as np
import pytest

from matchcake import (
    MatchgateStandardParams
)
from ..configs import N_RANDOM_TESTS_PER_CASE, TEST_SEED, ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "params,expected",
    [
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=-1.0, x=1.0, y=1.0, z=1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, -1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=-1.0, y=1.0, z=-1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, -1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=-1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
    ],
)
def test_standard_params_to_matrix(
        params: MatchgateStandardParams,
        expected: np.ndarray,
):
    matrix = params.to_matrix()
    np.testing.assert_allclose(matrix.squeeze(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@pytest.mark.parametrize(
    "matrix,params",
    [
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=1.0),
        ),
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, -1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=-1.0, x=1.0, y=1.0, z=1.0),
        ),
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, -1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=-1.0, y=1.0, z=-1.0),
        ),
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=-1.0),
        ),
    ],
)
def test_standard_params_from_matrix(matrix, params):
    params_ = MatchgateStandardParams.from_matrix(matrix)
    assert params_ == params

