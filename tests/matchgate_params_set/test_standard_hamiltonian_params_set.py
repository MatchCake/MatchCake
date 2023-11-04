import pytest
from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from msim import utils
import numpy as np
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.mark.parametrize(
    "std_ham_params0,std_ham_params1",
    [
        (
            MatchgateStandardHamiltonianParams.from_numpy(vector),
            MatchgateStandardHamiltonianParams.from_numpy(vector),
        )
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 8)
    ],
)
def test_standard_ham_from_standard_ham_params(
        std_ham_params0: MatchgateStandardHamiltonianParams,
        std_ham_params1: MatchgateStandardHamiltonianParams
):
    standard_params1_ = MatchgateStandardHamiltonianParams.parse_from_params(std_ham_params0)
    assert standard_params1_ == std_ham_params1


@pytest.mark.parametrize(
    "std_params,std_ham_params",
    [
        (
            MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_parse_from_standard_params(
        std_params: MatchgateStandardParams,
        std_ham_params: MatchgateStandardHamiltonianParams
):
    std_ham_params_ = MatchgateStandardHamiltonianParams.parse_from_standard_params(std_params)
    assert std_ham_params_ == std_ham_params

    std_ham_params__ = MatchgateStandardHamiltonianParams.parse_from_params(std_params)
    assert std_ham_params__ == std_ham_params


@pytest.mark.parametrize(
    "polar_params,std_ham_params",
    [
        (
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_parse_from_polar_params(
        polar_params: MatchgatePolarParams,
        std_ham_params: MatchgateStandardHamiltonianParams
):
    std_ham_params_ = MatchgateStandardHamiltonianParams.parse_from_polar_params(polar_params)
    assert std_ham_params_ == std_ham_params

    std_ham_params__ = MatchgateStandardHamiltonianParams.parse_from_params(polar_params)
    assert std_ham_params__ == std_ham_params


@pytest.mark.parametrize(
    "ham_params,std_ham_params",
    [
        (
            MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_parse_from_hamiltonian_coeffs_params(
        ham_params: MatchgateHamiltonianCoefficientsParams,
        std_ham_params: MatchgateStandardHamiltonianParams
):
    std_ham_params_ = MatchgateStandardHamiltonianParams.parse_from_hamiltonian_coefficients_params(ham_params)
    assert std_ham_params_ == std_ham_params

    std_ham_params__ = MatchgateStandardHamiltonianParams.parse_from_params(ham_params)
    assert std_ham_params__ == std_ham_params


@pytest.mark.parametrize(
    "composed_ham_params,std_ham_params",
    [
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
        ),
    ],
)
def test_parse_from_composed_hamiltonian_params(
        composed_ham_params: MatchgateComposedHamiltonianParams,
        std_ham_params: MatchgateStandardHamiltonianParams
):
    std_ham_params_ = MatchgateStandardHamiltonianParams.parse_from_composed_hamiltonian_params(composed_ham_params)
    assert std_ham_params_ == std_ham_params

    std_ham_params__ = MatchgateStandardHamiltonianParams.parse_from_params(composed_ham_params)
    assert std_ham_params__ == std_ham_params


@pytest.mark.parametrize(
    "params",
    [
        MatchgateStandardHamiltonianParams.from_numpy(vector)
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 8)
    ],
)
def test_parse_to_standard_back_and_forth(
        params: MatchgateStandardHamiltonianParams,
):
    standard_params = MatchgateStandardParams.parse_from_params(params)
    params_ = MatchgateStandardHamiltonianParams.parse_from_params(standard_params)
    assert params_ == params


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
    ],
)
def test_parse_from_hamiltonian_coeffs_with_slow_method(
        params: MatchgateHamiltonianCoefficientsParams,
):
    std_h_params = MatchgateStandardHamiltonianParams.parse_from_params(params)

    hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(std_h_params.to_matrix())
    elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
    params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    params_ = MatchgateHamiltonianCoefficientsParams.from_numpy(params_arr)
    assert params_ == std_h_params





