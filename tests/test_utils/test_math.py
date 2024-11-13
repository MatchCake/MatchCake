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
HIGH_VALUE = 2*np.pi
LOW_PRECISION_TERMS = 10
MEDIUM_PRECISION_TERMS = 18
HIGH_PRECISION_TERMS = 26


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-HIGH_VALUE, HIGH_VALUE), n_terms)
        for n_terms in range(LOW_PRECISION_TERMS, LOW_PRECISION_TERMS+N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_exp_taylor_series_approx(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(out, target, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-HIGH_VALUE, HIGH_VALUE), n_terms)
        for n_terms in range(MEDIUM_PRECISION_TERMS, MEDIUM_PRECISION_TERMS+N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_exp_taylor_series_medium_precision(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(
        out, target,
        atol=(ATOL_SCALAR_COMPARISON + ATOL_APPROX_COMPARISON) / 2,
        rtol=(RTOL_SCALAR_COMPARISON + RTOL_APPROX_COMPARISON) / 2,
    )


@pytest.mark.parametrize(
    "x, n_terms",
    [
        (np.random.uniform(-HIGH_VALUE, HIGH_VALUE), n_terms)
        for n_terms in range(HIGH_PRECISION_TERMS, HIGH_PRECISION_TERMS+N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_exp_taylor_series_high_precision(x, n_terms):
    target = np.exp(x)
    out = utils.math.exp_taylor_series(x, n_terms)
    np.testing.assert_allclose(out, target, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)


@pytest.mark.parametrize(
    "probs_shape",
    [
        probs_shape
        for probs_shape in [(10, ), (32, 10, )]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_random_index(probs_shape):
    n = 1000 * probs_shape[-1]
    probs = np.random.uniform(0, 1, probs_shape)
    probs_normed = probs / probs.sum(axis=-1, keepdims=True)
    indexes = utils.math.random_index(probs_normed, n=n)
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=probs.shape[-1]),
        axis=0,
        arr=indexes
    ).T
    probs_estimate = counts / indexes.shape[0]
    probs_estimate_normed = probs_estimate / probs_estimate.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        probs_estimate_normed, probs_normed,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "probs_shape",
    [
        probs_shape
        for probs_shape in [(4, 3, 2), (5, 4, 3)]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_random_index_3d_none(probs_shape):
    n = 128 * np.prod(probs_shape)

    probs = np.random.uniform(0, 1, probs_shape)
    probs_normed = probs / probs.sum(axis=-1, keepdims=True)
    indexes = np.stack([utils.math.random_index(probs_normed, n=None) for _ in range(n)], axis=0)
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=probs.shape[-1]),
        axis=0,
        arr=indexes.reshape(indexes.shape[0], -1)
    ).T.reshape(probs.shape)
    probs_estimate = counts / indexes.shape[0]
    probs_estimate_normed = probs_estimate / probs_estimate.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        probs_estimate_normed, probs_normed,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
