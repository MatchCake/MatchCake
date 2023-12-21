import numpy as np
import pytest

from msim import (
    MatchgatePolarParams
)
from ..configs import N_RANDOM_TESTS_PER_CASE, TEST_SEED, ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "r0, r1, theta0, theta1, theta2, theta3",
    [
        tuple(np.random.rand(6))
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_matchgate_polar_params_constructor(r0, r1, theta0, theta1, theta2, theta3):
    matchgate_params = MatchgatePolarParams(
        r0=r0,
        r1=r1,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
    )
    np.testing.assert_allclose(
        matchgate_params.r0, r0,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )
    np.testing.assert_allclose(
        matchgate_params.r1, r1,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )
    np.testing.assert_allclose(
        matchgate_params.theta0, theta0,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )
    np.testing.assert_allclose(
        matchgate_params.theta1, theta1,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )
    np.testing.assert_allclose(
        matchgate_params.theta2, theta2,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )
    np.testing.assert_allclose(
        matchgate_params.theta3, theta3,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )
    

@pytest.mark.parametrize(
    "params, batch_size",
    [
        (np.random.rand(b, MatchgatePolarParams.N_PARAMS), b)
        for b in [1, 2, 16, ]
    ]
)
def test_matchgate_polar_params_constructor_batch(params, batch_size):
    matchgate_params = MatchgatePolarParams(params)
    np.testing.assert_equal(
        matchgate_params.to_numpy().shape, (batch_size, MatchgatePolarParams.N_PARAMS)
    )

