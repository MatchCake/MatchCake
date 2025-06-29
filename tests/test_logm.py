import numpy as np
import pytest
from pennylane.math import expm

from matchcake.utils.math import logm

from .configs import (
    ATOL_MATRIX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)

NEAR_ZERO, NEAR_INF = 1e-16, 1e0


@pytest.mark.parametrize(
    "x",
    [np.random.uniform(-NEAR_INF, NEAR_INF, size=(4, 4)) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_logm(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [np.random.uniform(-NEAR_INF, NEAR_INF, size=(4, 4)) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_logm_torch(x):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    x = torch.from_numpy(x)
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [np.random.uniform(-NEAR_ZERO, NEAR_ZERO, size=(4, 4)) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_logm_near_0(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [np.random.uniform(NEAR_INF / 10, NEAR_INF, size=(4, 4)) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_logm_near_inf(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [np.random.uniform(-NEAR_INF, -NEAR_INF / 10, size=(4, 4)) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_logm_near_minf(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_INF, NEAR_INF, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_logm_complex(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_ZERO, NEAR_ZERO, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_logm_complex_near_0(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(NEAR_INF / 10, NEAR_INF, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_logm_complex_near_inf(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_INF, -NEAR_INF / 10, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_logm_complex_near_minf(x):
    np.testing.assert_allclose(
        x,
        expm(logm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        x,
        logm(expm(x)),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
