import numpy as np
import pytest

from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
    utils,
)

np.random.seed(42)


@pytest.mark.parametrize(
    "hamiltonian_params0,hamiltonian_params1",
    [
        (
            MatchgateHamiltonianCoefficientsParams.from_numpy(vector),
            MatchgateHamiltonianCoefficientsParams.from_numpy(vector),
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_hamiltonian_from_hamiltonian_params(
        hamiltonian_params0: MatchgateHamiltonianCoefficientsParams,
        hamiltonian_params1: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params1_ = MatchgateHamiltonianCoefficientsParams.parse_from_params(hamiltonian_params0)
    assert hamiltonian_params1_ == hamiltonian_params1


@pytest.mark.parametrize(
    "polar_params,hamiltonian_params",
    [
        (
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_parse_from_polar_params(
        polar_params: MatchgatePolarParams,
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_polar_params(polar_params)
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(polar_params)
    assert hamiltonian_params__ == hamiltonian_params


@pytest.mark.parametrize(
    "composed_hamiltonian_params,hamiltonian_params",
    [
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_parse_from_composed_hamiltonian_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_composed_hamiltonian_params(
        composed_hamiltonian_params
    )
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(composed_hamiltonian_params)
    assert hamiltonian_params__ == hamiltonian_params


@pytest.mark.parametrize(
    "standard_params,hamiltonian_params",
    [
        (
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_parse_from_standard_params(
        standard_params: MatchgateStandardParams,
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams
):
    hamiltonian_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_standard_params(standard_params)
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(standard_params)
    assert hamiltonian_params__ == hamiltonian_params


@pytest.mark.parametrize(
    "std_ham_params,ham_params",
    [
        (
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_parse_from_standard_hamiltonian_params(
        std_ham_params: MatchgateStandardHamiltonianParams,
        ham_params: MatchgateHamiltonianCoefficientsParams
):
    ham_params_ = MatchgateHamiltonianCoefficientsParams.parse_from_standard_hamiltonian_params(std_ham_params)
    assert ham_params_ == ham_params

    ham_params__ = MatchgateHamiltonianCoefficientsParams.parse_from_params(std_ham_params)
    assert ham_params__ == ham_params


@pytest.mark.parametrize(
    "params",
    [
        (
            MatchgateComposedHamiltonianParams.from_numpy(vector)
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_parse_from_composed_hamiltonian_params_is_real(
        params: MatchgateComposedHamiltonianParams,
):
    ham_params = MatchgateHamiltonianCoefficientsParams.parse_from_composed_hamiltonian_params(params)
    assert utils.check_if_imag_is_zero(ham_params.to_numpy()), "Hamiltonian params must be real."


@pytest.mark.parametrize(
    "params",
    [
        (
            MatchgatePolarParams.from_numpy(vector)
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_parse_from_polar_params_is_real(
        params: MatchgatePolarParams,
):
    ham_params = MatchgateHamiltonianCoefficientsParams.parse_from_polar_params(params)
    assert utils.check_if_imag_is_zero(ham_params.to_numpy()), "Hamiltonian params must be real."
