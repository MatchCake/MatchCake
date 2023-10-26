import pytest
from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianParams,
    MatchgateComposedHamiltonianParams
)
import numpy as np


@pytest.mark.parametrize(
    "r0, r1, theta0, theta1, theta2, theta3, theta4",
    [
        tuple(np.random.rand(7))
        for _ in range(100)
    ]
)
def test_matchgate_polar_params_constructor(r0, r1, theta0, theta1, theta2, theta3, theta4):
    matchgate_params = MatchgatePolarParams(
        r0=r0,
        r1=r1,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        theta4=theta4
    )
    assert np.isclose(matchgate_params.r0, r0)
    assert np.isclose(matchgate_params.r1, r1)
    assert np.isclose(matchgate_params.theta0, theta0)
    assert np.isclose(matchgate_params.theta1, theta1)
    assert np.isclose(matchgate_params.theta2, theta2)
    assert np.isclose(matchgate_params.theta3, theta3)
    assert np.isclose(matchgate_params.theta4, theta4)








