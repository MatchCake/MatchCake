import pytest
from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams
)
import numpy as np

np.random.seed(42)


@pytest.mark.parametrize(
    "r0, r1, theta0, theta1, theta2, theta3",
    [
        tuple(np.random.rand(6))
        for _ in range(100)
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


@pytest.mark.parametrize(
    "polar_params0,polar_params1",
    [
        (
                MatchgatePolarParams.from_numpy(vector),
                MatchgatePolarParams.from_numpy(vector),
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_polar_to_polar_params(
        polar_params0: MatchgatePolarParams,
        polar_params1: MatchgatePolarParams
):
    polar_params1_ = MatchgatePolarParams.parse_from_params(polar_params0)
    assert polar_params1_ == polar_params1


@pytest.mark.parametrize(
    "polar_params,standard_params",
    [
        (
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_polar_to_standard_params(
        polar_params: MatchgatePolarParams,
        standard_params: MatchgateStandardParams
):
    standard_params_ = MatchgateStandardParams.parse_from_polar_params(polar_params)
    assert standard_params_ == standard_params
    
    standard_params__ = MatchgateStandardParams.parse_from_params(polar_params)
    assert standard_params__ == standard_params


@pytest.mark.parametrize(
    "polar_params,hamiltonian_params",
    [
        (
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_polar_to_hamiltonian_params(
        polar_params: MatchgatePolarParams,
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_polar_params(polar_params)
    assert hamiltonian_params_ == hamiltonian_params
    
    hamiltonian_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(polar_params)
    assert hamiltonian_params__ == hamiltonian_params


@pytest.mark.parametrize(
    "polar_params,composed_hamiltonian_params",
    [
        (
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_polar_to_hamiltonian_params(
        polar_params: MatchgatePolarParams,
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params_ = MatchgateComposedHamiltonianParams.parse_from_polar_params(polar_params)
    assert composed_hamiltonian_params_ == composed_hamiltonian_params
    
    composed_hamiltonian_params__ = MatchgateComposedHamiltonianParams.parse_from_params(polar_params)
    assert composed_hamiltonian_params__ == composed_hamiltonian_params
