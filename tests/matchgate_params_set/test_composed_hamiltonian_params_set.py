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
def test_composed_hamiltonian_from_composed_hamiltonian_params(
        composed_hamiltonian_params0: MatchgateComposedHamiltonianParams,
        composed_hamiltonian_params1: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params1_ = MatchgateComposedHamiltonianParams.parse_from_params(composed_hamiltonian_params0)
    assert composed_hamiltonian_params1_ == composed_hamiltonian_params1


@pytest.mark.parametrize(
    "standard_params,composed_hamiltonian_params",
    [
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_parse_from_standard_params(
        standard_params: MatchgateStandardParams,
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params_ = MatchgateComposedHamiltonianParams.parse_from_standard_params(standard_params)
    assert composed_hamiltonian_params_ == composed_hamiltonian_params

    composed_hamiltonian_params__ = MatchgateComposedHamiltonianParams.parse_from_params(standard_params)
    assert composed_hamiltonian_params__ == composed_hamiltonian_params


@pytest.mark.parametrize(
    "polar_params,composed_hamiltonian_params",
    [
        (
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_parse_from_polar_params(
        polar_params: MatchgatePolarParams,
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params_ = MatchgateComposedHamiltonianParams.parse_from_polar_params(polar_params)
    assert composed_hamiltonian_params_ == composed_hamiltonian_params

    composed_hamiltonian_params__ = MatchgateComposedHamiltonianParams.parse_from_params(polar_params)
    assert composed_hamiltonian_params__ == composed_hamiltonian_params


@pytest.mark.parametrize(
    "hamiltonian_params,composed_hamiltonian_params",
    [
        (
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_parse_from_hamiltonian_params(
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams,
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params_ = MatchgateComposedHamiltonianParams.parse_from_hamiltonian_params(hamiltonian_params)
    assert composed_hamiltonian_params_ == composed_hamiltonian_params

    composed_hamiltonian_params__ = MatchgateComposedHamiltonianParams.parse_from_params(hamiltonian_params)
    assert composed_hamiltonian_params__ == composed_hamiltonian_params


@pytest.mark.parametrize(
    "std_ham_params,composed_ham_params",
    [
        (
                MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_parse_from_standard_hamiltonian_params(
        std_ham_params: MatchgateStandardHamiltonianParams,
        composed_ham_params: MatchgateComposedHamiltonianParams
):
    composed_ham_params_ = MatchgateComposedHamiltonianParams.parse_from_standard_hamiltonian_params(std_ham_params)
    assert composed_ham_params_ == composed_ham_params

    composed_ham_params__ = MatchgateComposedHamiltonianParams.parse_from_params(std_ham_params)
    assert composed_ham_params__ == composed_ham_params


@pytest.mark.parametrize(
    "params",
    [
        (
            MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_parse_from_standard_hamiltonian_params_is_real(
        params: MatchgateHamiltonianCoefficientsParams,
):
    composed_ham_params = MatchgateComposedHamiltonianParams.parse_from_hamiltonian_params(params)
    assert utils.check_if_imag_is_zero(composed_ham_params.to_numpy()), "Composed Hamiltonian params must be real."


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
    composed_ham_params = MatchgateComposedHamiltonianParams.parse_from_polar_params(params)
    assert utils.check_if_imag_is_zero(composed_ham_params.to_numpy()), "Composed Hamiltonian params must be real."

