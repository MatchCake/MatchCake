import numpy as np
import pytest

from msim import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_SCALAR_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)

np.random.seed(TEST_SEED)


def gen_skew_symmetric_matrix_and_det(n, batch_size=None):
    if batch_size is None:
        matrix = np.random.rand(n, n)
    else:
        matrix = np.random.rand(batch_size, n, n)
    matrix = matrix - np.einsum("...ij->...ji", matrix)
    return matrix, np.linalg.det(matrix)


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i)
        for i in range(2, N_RANDOM_TESTS_PER_CASE+2)
        for mth in ["P", "det"]
    ]
)
def test_pfaffian_ltl(matrix, det):
    pf = utils.pfaffian_ltl(matrix)
    np.testing.assert_allclose(pf ** 2, det, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i, batch_size=5)
        for i in range(2, N_RANDOM_TESTS_PER_CASE+2)
    ]
)
def test_batch_pfaffian_ltl(matrix, det):
    pf = utils._pfaffian.batch_pfaffian_ltl(matrix)
    np.testing.assert_allclose(pf ** 2, det, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

