import numpy as np
import pytest

from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
    utils
)
from msim.utils import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_I,
)
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


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
    assert np.isclose(matchgate_params.r0, r0)
    assert np.isclose(matchgate_params.r1, r1)
    assert np.isclose(matchgate_params.theta0, theta0)
    assert np.isclose(matchgate_params.theta1, theta1)
    assert np.isclose(matchgate_params.theta2, theta2)
    assert np.isclose(matchgate_params.theta3, theta3)




