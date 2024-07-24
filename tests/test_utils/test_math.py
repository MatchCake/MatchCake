import numpy as np
import pytest

from matchcake import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-1, 1), n_terms)
        for n_terms in range(2, 2+N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_exp_taylor_series_approx(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(out, target, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-1, 1), n_terms)
        for n_terms in range(10, 10+N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_exp_taylor_series_high_precision(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(out, target, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
