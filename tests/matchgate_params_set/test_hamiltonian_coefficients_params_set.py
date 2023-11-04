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
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.mark.parametrize(
    "hamiltonian_params0,hamiltonian_params1",
    [
        (
            MatchgateHamiltonianCoefficientsParams.from_numpy(vector),
            MatchgateHamiltonianCoefficientsParams.from_numpy(vector),
        )
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
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
        (
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=1.0, m_y=1.0, m_z=1.0),
            MatchgateHamiltonianCoefficientsParams(h0=2.0, h1=2.0, h2=0.0, h3=2.0, h4=0.0, h5=0.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=2.0, m_y=2.0, m_z=2.0),
            MatchgateHamiltonianCoefficientsParams(h0=3.0, h1=3.0, h2=-1.0, h3=3.0, h4=-1.0, h5=-1.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=2.0, n_y=2.0, n_z=2.0, m_x=1.0, m_y=1.0, m_z=1.0),
            MatchgateHamiltonianCoefficientsParams(h0=3.0, h1=3.0, h2=1.0, h3=3.0, h4=1.0, h5=1.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=1.0, n_y=1.0, n_z=1.0, m_x=0.0, m_y=0.0, m_z=0.0),
            MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
        ),
        (
            MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=1.0, m_y=1.0, m_z=1.0),
            MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=-1.0, h3=1.0, h4=-1.0, h5=-1.0),
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
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
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
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 7)
    ],
)
def test_parse_from_polar_params_is_real(
        params: MatchgatePolarParams,
):
    ham_params = MatchgateHamiltonianCoefficientsParams.parse_from_polar_params(params)
    assert utils.check_if_imag_is_zero(ham_params.to_numpy()), "Hamiltonian params must be real."


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
    ],
)
def test_parse_to_standard_back_and_forth(
        params: MatchgateHamiltonianCoefficientsParams,
):
    standard_params = MatchgateStandardParams.parse_from_params(params)
    params_ = MatchgateHamiltonianCoefficientsParams.parse_from_params(standard_params)
    assert params_ == params


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
    ],
)
def test_parse_to_standard_hamiltonian_back_and_forth(
        params: MatchgateHamiltonianCoefficientsParams,
):
    standard_h_params = MatchgateStandardHamiltonianParams.parse_from_params(params)
    params_ = MatchgateHamiltonianCoefficientsParams.parse_from_params(standard_h_params)
    assert params_ == params


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
        MatchgateHamiltonianCoefficientsParams(h0=-1.0, h1=-1.0, h2=-1.0, h3=1.0, h4=1.0, h5=1.0),

    ]
    +
    [
        MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
    ],
)
def test_parse_to_composed_hamiltonian_back_and_forth(
        params: MatchgateHamiltonianCoefficientsParams,
):
    composed_h_params = MatchgateComposedHamiltonianParams.parse_from_hamiltonian_coefficients_params(params)
    params_ = MatchgateHamiltonianCoefficientsParams.parse_from_composed_hamiltonian_params(composed_h_params)
    assert params_ == params


@pytest.mark.parametrize(
    "vector",
    [
        np.random.rand(6)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_from_numpy(
        vector: np.ndarray,
):
    params = MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
    assert np.allclose(params.to_numpy(), vector)

    params_ = MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
    assert params_ == params

    params__ = MatchgateHamiltonianCoefficientsParams(
        h0=vector[0],
        h1=vector[1],
        h2=vector[2],
        h3=vector[3],
        h4=vector[4],
        h5=vector[5],
        backend="numpy",
    )
    assert params__ == params

