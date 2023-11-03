import numpy as np
import pytest

from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams, utils
)

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
def test_polar_from_polar_params(
        polar_params0: MatchgatePolarParams,
        polar_params1: MatchgatePolarParams
):
    polar_params1_ = MatchgatePolarParams.parse_from_params(polar_params0)
    assert polar_params1_ == polar_params1


@pytest.mark.parametrize(
    "standard_params,polar_params",
    [
        (
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0)
        ),
    ],
)
def test_parse_from_standard_params(standard_params: MatchgateStandardParams, polar_params: MatchgatePolarParams):
    polar_params_ = MatchgatePolarParams.parse_from_standard_params(standard_params)
    assert polar_params_ == polar_params

    polar_params__ = MatchgatePolarParams.parse_from_params(standard_params)
    assert polar_params__ == polar_params


@pytest.mark.parametrize(
    "composed_hamiltonian_params,polar_params",
    [
        (
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
    ],
)
def test_parse_from_composed_hamiltonian_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        polar_params: MatchgatePolarParams
):
    polar_params_ = MatchgatePolarParams.parse_from_composed_hamiltonian_params(composed_hamiltonian_params)
    assert polar_params_ == polar_params

    polar_params__ = MatchgatePolarParams.parse_from_params(composed_hamiltonian_params)
    assert polar_params__ == polar_params


@pytest.mark.parametrize(
    "hamiltonian_params,polar_params",
    [
        (
                MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
    ],
)
def test_parse_from_hamiltonian_params(
        hamiltonian_params: MatchgateHamiltonianCoefficientsParams,
        polar_params: MatchgatePolarParams
):
    polar_params_ = MatchgatePolarParams.parse_from_hamiltonian_params(hamiltonian_params)
    assert polar_params_ == polar_params

    polar_params__ = MatchgatePolarParams.parse_from_params(hamiltonian_params)
    assert polar_params__ == polar_params


@pytest.mark.parametrize(
    "std_ham_params,polar_params",
    [
        (
            MatchgateStandardHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0),
            MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0)
        ),
    ],
)
def test_parse_from_standard_hamiltonian_params(
        std_ham_params: MatchgateStandardHamiltonianParams,
        polar_params: MatchgatePolarParams
):
    polar_params_ = MatchgatePolarParams.parse_from_standard_hamiltonian_params(std_ham_params)
    assert polar_params_ == polar_params

    polar_params__ = MatchgatePolarParams.parse_from_params(std_ham_params)
    assert polar_params__ == polar_params


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
    polar_params = MatchgatePolarParams.parse_from_hamiltonian_params(params)
    assert utils.check_if_imag_is_zero(polar_params.to_numpy()), "Polar params must be real."


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
    polar_params = MatchgatePolarParams.parse_from_composed_hamiltonian_params(params)
    assert utils.check_if_imag_is_zero(polar_params.to_numpy()), "Polar params must be real."
