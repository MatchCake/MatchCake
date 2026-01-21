import numpy as np
import pytest
import torch
from pennylane.math import expm
from torch.autograd import gradcheck

from matchcake.utils.logm import logm, torch_logm
from matchcake.utils.torch_utils import to_tensor

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
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


class TestLogm:
    def test_forward(self):
        x = np.linspace(0, 100, num=6**2).reshape(6, 6)
        x = to_tensor(x)
        res = expm(logm(x))
        torch.testing.assert_close(res, x, check_dtype=False)

    def test_backward(self):
        pytest.skip(
            "Should work but doesn't. See https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620."
        )
        x = np.linspace(0, 1, num=6**2).reshape(6, 6)
        torch_x = torch.from_numpy(x).requires_grad_().to(torch.complex128)
        assert gradcheck(logm, (torch_x,))
