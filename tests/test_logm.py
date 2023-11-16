import pytest
from .configs import N_RANDOM_TESTS_PER_CASE
import numpy as np
from scipy.linalg import expm, logm


NEAR_ZERO, NEAR_INF = 1e-16, 1e0


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_INF, NEAR_INF, size=(4, 4))
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_ZERO, NEAR_ZERO, size=(4, 4))
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_near_0(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(NEAR_INF/10, NEAR_INF, size=(4, 4))
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_near_inf(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_INF, -NEAR_INF/10, size=(4, 4))
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_near_minf(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_INF, NEAR_INF, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_complex(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_ZERO, NEAR_ZERO, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_complex_near_0(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )


@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(NEAR_INF/10, NEAR_INF, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_complex_near_inf(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )
    

@pytest.mark.parametrize(
    "x",
    [
        np.random.uniform(-NEAR_INF, -NEAR_INF/10, size=(4, 4, 2)).view(np.complex128).reshape(4, 4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_logm_complex_near_minf(x):
    assert np.allclose(x, expm(logm(x))), (
        f"expm(logm(x)) is not equal to x. x = \n{x}\nexpm(logm(x)) = \n{expm(logm(x))}"
    )
    assert np.allclose(x, logm(expm(x))), (
        f"logm(expm(x)) is not equal to x. x = \n{x}\nlogm(expm(x)) = \n{logm(expm(x))}"
    )
