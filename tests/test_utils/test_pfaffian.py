from functools import partial

import numpy as np
import pytest

from matchcake import utils

from .. import get_slow_test_mark
from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)
MIN_MATRIX_SIZE = 2
MAX_MATRIX_SIZE = 6
BATCH_SIZE = 3
RECOMMENDED_METHODS = ["det", "bLTL", "bH", "PfaffianFDBPf"]


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
        gen_skew_symmetric_matrix_and_det(i, batch_size=None)
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_get_skew_symmetric_matrix_and_det(matrix, det):
    try:
        from pfapack.pfaffian import pfaffian
    except ImportError:
        pytest.skip("pfapack is not installed.")
    pf = pfaffian(matrix)
    np.testing.assert_allclose(pf**2, det, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)


@pytest.mark.parametrize(
    "matrix, det, mth",
    [
        (*gen_skew_symmetric_matrix_and_det(i, batch_size=batch_size), mth)
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for mth in RECOMMENDED_METHODS
        for batch_size in [None, BATCH_SIZE]
    ],
)
def test_pfaffian_methods(matrix, det, mth):
    pf = utils.pfaffian(matrix, method=mth)
    np.testing.assert_allclose(pf**2, det, atol=10 * ATOL_SCALAR_COMPARISON, rtol=10 * RTOL_SCALAR_COMPARISON)


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i, batch_size=BATCH_SIZE)
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_batch_pfaffian_ltl(matrix, det):
    pf = utils._pfaffian.batch_pfaffian_ltl(matrix)
    np.testing.assert_allclose(pf**2, det, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i, batch_size=BATCH_SIZE)
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_pfaffian_bltl(matrix, det):
    pf = utils.pfaffian(matrix, method="bLTL")
    np.testing.assert_allclose(pf**2, det, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i, batch_size=batch_size)
        for batch_size in [None, 1]
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_pfaffian_bltl_single_item(matrix, det):
    pf = utils.pfaffian(matrix, method="bLTL")
    np.testing.assert_allclose(pf**2, det, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "n, batch_size, mth",
    [
        (i, batch_size, mth)
        for i in np.linspace(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE, num=N_RANDOM_TESTS_PER_CASE, dtype=int)
        for mth in RECOMMENDED_METHODS
        for batch_size in [None, BATCH_SIZE]
    ],
)
def test_pfaffian_methods_grads(n, batch_size, mth):
    try:
        import torch
        from torch.autograd import gradcheck
    except ImportError:
        pytest.skip("PyTorch is not installed.")

    if batch_size is None:
        np_matrix = np.random.rand(n, n)
    else:
        np_matrix = np.random.rand(batch_size, n, n)
    np_matrix = np_matrix - np.einsum("...ij->...ji", np_matrix)
    torch_matrix = torch.from_numpy(np_matrix).requires_grad_()
    func = partial(utils.pfaffian, method=mth)
    assert gradcheck(
        func,
        (torch_matrix,),
        atol=ATOL_APPROX_COMPARISON,
        rtol=10 * RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i, batch_size=BATCH_SIZE)
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_pfaffian_bh(matrix, det):
    pf = utils.pfaffian(matrix, method="bH")
    np.testing.assert_allclose(pf**2, det, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@pytest.mark.parametrize(
    "matrix, det",
    [
        gen_skew_symmetric_matrix_and_det(i, batch_size=batch_size)
        for batch_size in [None, 1]
        for i in range(MIN_MATRIX_SIZE, MAX_MATRIX_SIZE + 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_pfaffian_bltl_single_item(matrix, det):
    pf = utils.pfaffian(matrix, method="bH")
    np.testing.assert_allclose(pf**2, det, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
