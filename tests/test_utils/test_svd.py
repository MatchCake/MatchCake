from functools import partial

import numpy as np
import pytest

import pennylane as qml

from matchcake import utils
from matchcake.utils.math import svd
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_SCALAR_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)



@pytest.mark.parametrize(
    "matrix",
    [
        np.random.rand(batch_size, size, size).squeeze()
        for size in range(2, 10)
        for batch_size in [1, 3]
    ]
)
def test_orthonormalize(matrix):
    u, s, v = svd(matrix)
    pred_matrix = np.einsum("...ik,...k,...kj->...ij", u, s, v)
    np.testing.assert_allclose(pred_matrix, matrix, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)