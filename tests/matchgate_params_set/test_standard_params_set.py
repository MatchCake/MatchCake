import numpy as np
import pytest

from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams
)

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
def test_standard_from_standard_params(
        standard_params0: MatchgateStandardParams,
        standard_params1: MatchgateStandardParams
):
    standard_params1_ = MatchgateStandardParams.parse_from_params(standard_params0)
    assert standard_params1_ == standard_params1


@pytest.mark.parametrize(
    "std_ham_params,std_params",
    [
        (
                MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_parse_from_standard_hamiltonian_params(
        std_ham_params: MatchgateStandardHamiltonianParams,
        std_params: MatchgateStandardParams
):
    std_params_ = MatchgateStandardParams.parse_from_standard_hamiltonian_params(std_ham_params)
    assert std_params_ == std_params

    std_params__ = MatchgateStandardParams.parse_from_params(std_params)
    assert std_params__ == std_params


@pytest.mark.parametrize(
    "polar_params,standard_params",
    [
        (
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_parse_from_polar_params(
        polar_params: MatchgatePolarParams,
        standard_params: MatchgateStandardParams
):
    standard_params_ = MatchgateStandardParams.parse_from_polar_params(polar_params)
    assert standard_params_ == standard_params

    standard_params__ = MatchgateStandardParams.parse_from_params(polar_params)
    assert standard_params__ == standard_params


@pytest.mark.parametrize(
    "hamiltonian_params,standard_params",
    [
        (
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_parse_from_hamiltonian_params(
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams,
        standard_params: MatchgateStandardParams
):
    standard_params_ = MatchgateStandardParams.parse_from_hamiltonian_params(hamiltonian_params)
    assert standard_params_ == standard_params

    standard_params__ = MatchgateStandardParams.parse_from_params(hamiltonian_params)
    assert standard_params__ == standard_params


@pytest.mark.parametrize(
    "composed_hamiltonian_params,standard_params",
    [
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_parse_from_composed_hamiltonian_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        standard_params: MatchgateStandardParams
):
    standard_params_ = MatchgateStandardParams.parse_from_composed_hamiltonian_params(composed_hamiltonian_params)
    assert standard_params_ == standard_params

    standard_params__ = MatchgateStandardParams.parse_from_params(composed_hamiltonian_params)
    assert standard_params__ == standard_params
