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
    "composed_hamiltonian_params0,composed_hamiltonian_params1",
    [
        (
                MatchgateComposedHamiltonianParams.from_numpy(vector),
                MatchgateComposedHamiltonianParams.from_numpy(vector),
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_composed_hamiltonian_to_composed_hamiltonian_params(
        composed_hamiltonian_params0: MatchgateComposedHamiltonianParams,
        composed_hamiltonian_params1: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params1_ = MatchgateComposedHamiltonianParams.parse_from_params(composed_hamiltonian_params0)
    assert composed_hamiltonian_params1_ == composed_hamiltonian_params1


@pytest.mark.parametrize(
    "composed_hamiltonian_params,standard_params",
    [
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_composed_hamiltonian_to_standard_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        standard_params: MatchgateStandardParams
):
    standard_params_ = MatchgateStandardParams.parse_from_composed_hamiltonian_params(composed_hamiltonian_params)
    assert standard_params_ == standard_params
    
    standard_params__ = MatchgateStandardParams.parse_from_params(composed_hamiltonian_params)
    assert standard_params__ == standard_params


@pytest.mark.parametrize(
    "composed_hamiltonian_params,polar_params",
    [
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
    ],
)
def test_composed_hamiltonian_to_polar_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        polar_params: MatchgatePolarParams
):
    polar_params_ = MatchgatePolarParams.parse_from_composed_hamiltonian_params(composed_hamiltonian_params)
    assert polar_params_ == polar_params
    
    polar_params__ = MatchgatePolarParams.parse_from_params(composed_hamiltonian_params)
    assert polar_params__ == polar_params


@pytest.mark.parametrize(
    "composed_hamiltonian_params,hamiltonian_params",
    [
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_composed_hamiltonian_to_hamiltonian_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_composed_hamiltonian_params(
        composed_hamiltonian_params
        )
    assert hamiltonian_params_ == hamiltonian_params
    
    hamiltonian_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(composed_hamiltonian_params)
    assert hamiltonian_params__ == hamiltonian_params
