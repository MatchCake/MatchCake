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
    "standard_params0,standard_params1",
    [
        (
                MatchgateStandardParams.from_numpy(vector),
                MatchgateStandardParams.from_numpy(vector),
        )
        for vector in np.random.rand(100, 8)
    ],
)
def test_standard_to_standard_params(
        standard_params0: MatchgateStandardParams,
        standard_params1: MatchgateStandardParams
):
    standard_params1_ = MatchgateStandardParams.parse_from_params(standard_params0)
    assert standard_params1_ == standard_params1


@pytest.mark.parametrize(
    "standard_params,polar_params",
    [
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0)
        ),
    ],
)
def test_standard_to_polar_params(standard_params: MatchgateStandardParams, polar_params: MatchgatePolarParams):
    polar_params_ = MatchgatePolarParams.parse_from_standard_params(standard_params)
    assert polar_params_ == polar_params

    polar_params__ = MatchgatePolarParams.parse_from_params(standard_params)
    assert polar_params__ == polar_params


@pytest.mark.parametrize(
    "standard_params,hamiltonian_params",
    [
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_standard_to_hamiltonian_params(
        standard_params: MatchgateStandardParams,
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_standard_params(standard_params)
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(standard_params)
    assert hamiltonian_params__ == hamiltonian_params


@pytest.mark.parametrize(
    "standard_params,composed_hamiltonian_params",
    [
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_standard_to_hamiltonian_params(
        standard_params: MatchgateStandardParams,
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params_ = MatchgateComposedHamiltonianParams.parse_from_standard_params(standard_params)
    assert composed_hamiltonian_params_ == composed_hamiltonian_params

    composed_hamiltonian_params__ = MatchgateComposedHamiltonianParams.parse_from_params(standard_params)
    assert composed_hamiltonian_params__ == composed_hamiltonian_params


