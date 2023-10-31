import pytest
from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianParams,
    MatchgateComposedHamiltonianParams
)
import numpy as np


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
                MatchgateHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_standard_to_hamiltonian_params(
        standard_params: MatchgateStandardParams,
        hamiltonian_params: MatchgateHamiltonianParams
):
    hamiltonian_params_ = MatchgateHamiltonianParams.parse_from_standard_params(standard_params)
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianParams.parse_from_params(standard_params)
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
                MatchgateHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_polar_to_hamiltonian_params(
        polar_params: MatchgatePolarParams,
        hamiltonian_params: MatchgateHamiltonianParams
):
    hamiltonian_params_ = MatchgateHamiltonianParams.parse_from_polar_params(polar_params)
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianParams.parse_from_params(polar_params)
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


@pytest.mark.parametrize(
    "hamiltonian_params0,hamiltonian_params1",
    [
        (
                MatchgateHamiltonianParams.from_numpy(vector),
                MatchgateHamiltonianParams.from_numpy(vector),
        )
        for vector in np.random.rand(100, 6)
    ],
)
def test_hamiltonian_to_hamiltonian_params(
        hamiltonian_params0: MatchgateHamiltonianParams,
        hamiltonian_params1: MatchgateHamiltonianParams
):
    hamiltonian_params1_ = MatchgateHamiltonianParams.parse_from_params(hamiltonian_params0)
    assert hamiltonian_params1_ == hamiltonian_params1


@pytest.mark.parametrize(
    "hamiltonian_params,standard_params",
    [
        (
                MatchgateHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
                MatchgateStandardParams(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
        ),
    ],
)
def test_hamiltonian_to_standard_params(
        hamiltonian_params: MatchgateHamiltonianParams,
        standard_params: MatchgateStandardParams
):
    standard_params_ = MatchgateStandardParams.parse_from_hamiltonian_params(hamiltonian_params)
    assert standard_params_ == standard_params

    standard_params__ = MatchgateStandardParams.parse_from_params(hamiltonian_params)
    assert standard_params__ == standard_params


@pytest.mark.parametrize(
    "hamiltonian_params,polar_params",
    [
        (
                MatchgateHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
                MatchgatePolarParams(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
        ),
    ],
)
def test_hamiltonian_to_polar_params(
        hamiltonian_params: MatchgateHamiltonianParams,
        polar_params: MatchgatePolarParams
):
    polar_params_ = MatchgatePolarParams.parse_from_hamiltonian_params(hamiltonian_params)
    assert polar_params_ == polar_params

    polar_params__ = MatchgatePolarParams.parse_from_params(hamiltonian_params)
    assert polar_params__ == polar_params


@pytest.mark.parametrize(
    "hamiltonian_params,composed_hamiltonian_params",
    [
        (
                MatchgateHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
                MatchgateComposedHamiltonianParams(n_x=0.0, n_y=0.0, n_z=0.0, m_x=0.0, m_y=0.0, m_z=0.0),
        ),
    ],
)
def test_hamiltonian_to_hamiltonian_params(
        hamiltonian_params: MatchgateHamiltonianParams,
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams
):
    composed_hamiltonian_params_ = MatchgateComposedHamiltonianParams.parse_from_hamiltonian_params(hamiltonian_params)
    assert composed_hamiltonian_params_ == composed_hamiltonian_params

    composed_hamiltonian_params__ = MatchgateComposedHamiltonianParams.parse_from_params(hamiltonian_params)
    assert composed_hamiltonian_params__ == composed_hamiltonian_params


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
                MatchgateHamiltonianParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        ),
    ],
)
def test_composed_hamiltonian_to_hamiltonian_params(
        composed_hamiltonian_params: MatchgateComposedHamiltonianParams,
        hamiltonian_params: MatchgateHamiltonianParams
):
    hamiltonian_params_ = MatchgateHamiltonianParams.parse_from_composed_hamiltonian_params(composed_hamiltonian_params)
    assert hamiltonian_params_ == hamiltonian_params

    hamiltonian_params__ = MatchgateHamiltonianParams.parse_from_params(composed_hamiltonian_params)
    assert hamiltonian_params__ == hamiltonian_params





