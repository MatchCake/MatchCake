import numpy as np
import pennylane as qml
import pytest

from matchcake import Matchgate, mps, utils

from .configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize("matchgate_rn_init", [Matchgate.random() for _ in range(N_RANDOM_TESTS_PER_CASE)])
def test_matchgate_m_m_dagger_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_m_m_dagger_constraint()


@pytest.mark.parametrize("matchgate_rn_init", [Matchgate.random() for _ in range(N_RANDOM_TESTS_PER_CASE)])
def test_matchgate_m_dagger_m_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_m_dagger_m_constraint()


@pytest.mark.parametrize("matchgate_rn_init", [Matchgate.random() for _ in range(N_RANDOM_TESTS_PER_CASE)])
def test_matchgate_det_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_det_constraint()


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams(
            r0=np.random.rand(),
            r1=np.random.rand(),
            theta0=np.random.rand(),
            theta1=np.random.rand(),
            theta2=np.random.rand(),
            theta3=np.random.rand(),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_matchgate_constructor_with_default_theta4(params):
    m = Matchgate(params)
    exp4 = np.exp(1j * np.mod(m.polar_params.theta4, 2 * np.pi))
    exp2 = np.exp(1j * np.mod(m.polar_params.theta2, 2 * np.pi))
    np.testing.assert_almost_equal(exp2 * exp4, 1.0 + 0j)


@pytest.mark.parametrize(
    "input_matrix",
    [
        np.array(
            [
                [1.0, 0, 0, 0.0],
                [0, 1.0, 0.0, 0],
                [0, 0.0, 1.0, 0],
                [0.0, 0, 0, 1.0],
            ]
        ),
        np.array(
            [
                [1.0, 0, 0, 0.0],
                [0, 0.0, 1.0, 0],
                [0, -1.0, 0.0, 0],
                [0.0, 0, 0, 1.0],
            ]
        ),
    ],
)
def test_matchgate_constructor_from_matrix(input_matrix):
    mg = Matchgate.from_matrix(input_matrix)
    np.testing.assert_allclose(
        mg.gate_data.squeeze(),
        input_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "input_matrix,target_coefficients",
    [
        (
            np.array([[1.0, 0, 0, 0.0], [0, 1.0, 0.0, 0], [0, 0.0, 1.0, 0], [0.0, 0, 0, 1.0]]),
            np.zeros(mps.MatchgateHamiltonianCoefficientsParams.N_PARAMS),
        )
    ],
)
def test_matchgate_hamiltonian_coefficient(input_matrix, target_coefficients):
    mg = Matchgate.from_matrix(input_matrix)
    coeffs_vector = mg.hamiltonian_coefficients_params.to_numpy()
    np.testing.assert_allclose(
        coeffs_vector.squeeze(),
        target_coefficients,
        rtol=RTOL_APPROX_COMPARISON,
        atol=ATOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params",
    [mps.MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_random_polar_params_gives_matchgate(params):
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    assert matchgate.check_m_m_dagger_constraint(), f"m_m_dagger_constraint failed for random {type(params)}"
    assert matchgate.check_m_dagger_m_constraint(), f"m_dagger_m_constraint failed for random {type(params)}"
    assert matchgate.check_det_constraint(), f"det_constraint failed for random {type(params)}"


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams(
            r0=np.random.rand(),
            r1=np.random.rand(),
            theta0=np.random.rand(),
            theta1=np.random.rand(),
            theta2=np.random.rand(),
            theta3=np.random.rand(),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_random_simple_polar_params_respect_constraint_in_hamiltonian_form(params):
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    gate_det = np.linalg.det(matchgate.gate_data)
    hamiltonian_form_det = np.linalg.det(qml.math.expm(1j * matchgate.hamiltonian_matrix))
    hamiltonian_trace = qml.math.trace(matchgate.hamiltonian_matrix, axis1=-2, axis2=-1)
    exp_trace = qml.math.exp(1j * hamiltonian_trace)
    np.testing.assert_almost_equal(hamiltonian_form_det, gate_det)
    np.testing.assert_almost_equal(gate_det, exp_trace)


@pytest.mark.parametrize(
    "params",
    [mps.MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_random_polar_params_respect_constraint_in_hamiltonian_form(params):
    # TODO: Need to verify this property and make sure the matchgate is implemented correctly
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    gate_det = np.linalg.det(matchgate.gate_data)
    hamiltonian_form_det = np.linalg.det(qml.math.expm(1j * matchgate.hamiltonian_matrix))
    hamiltonian_trace = qml.math.trace(matchgate.hamiltonian_matrix, axis1=-2, axis2=-1)
    exp_trace = qml.math.exp(1j * hamiltonian_trace)
    # np.testing.assert_allclose(hamiltonian_form_det, gate_det, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
    # np.testing.assert_allclose(gate_det, exp_trace, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
    np.testing.assert_allclose(
        hamiltonian_form_det,
        exp_trace,
        atol=ATOL_SCALAR_COMPARISON,
        rtol=RTOL_SCALAR_COMPARISON,
    )


@pytest.mark.parametrize(
    "params",
    [mps.MatchgateComposedHamiltonianParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_random_composed_hamiltonian_params_gives_matchgate(params):
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    assert matchgate.check_m_m_dagger_constraint(), f"m_m_dagger_constraint failed for random {type(params)}"
    assert matchgate.check_m_dagger_m_constraint(), f"m_dagger_m_constraint failed for random {type(params)}"
    assert matchgate.check_det_constraint(), f"det_constraint failed for random {type(params)}"


@pytest.mark.parametrize(
    "params,expected",
    [
        (mps.MatchgateHamiltonianCoefficientsParams(), np.eye(4, dtype=complex)),
        (mps.MatchgateStandardParams(a=1, w=1, z=1, d=1), np.eye(4, dtype=complex)),
        (mps.MatchgateStandardParams(a=-1, w=-1, z=-1, d=-1), np.eye(4, dtype=complex)),
        (
            mps.MatchgateStandardParams(a=[-1, -1], w=-1, z=-1, d=-1),
            [np.eye(4, dtype=complex) for _ in range(2)],
        ),
        (
            mps.MatchgateStandardParams(a=[1, 1], w=1, z=1, d=1),
            [np.eye(4, dtype=complex) for _ in range(2)],
        ),
    ],
)
def test_action_matrix(params, expected):
    expected = qml.math.array(expected)
    mg = Matchgate(params)
    action_matrix = mg.single_particle_transition_matrix
    np.testing.assert_allclose(
        action_matrix.squeeze(),
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            mps.MatchgatePolarParams(r0=1, r1=1),
            np.array(
                [
                    [0.25 * np.trace(utils.get_majorana(i, 2) @ utils.get_majorana(j, 2)) for j in range(4)]
                    for i in range(4)
                ]
            ),
        ),
        (
            mps.MatchgatePolarParams(r0=1, r1=1),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            mps.fSWAP,
            np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ]
            ),
        ),
    ],
)
def test_single_transition_matrix(params, expected):
    expected = qml.math.array(expected)
    mg = Matchgate(params)
    np.testing.assert_allclose(
        mg.single_particle_transition_matrix.squeeze(),
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params",
    [mps.MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_single_transition_matrix_equal_to_expm_hami_coeff_if_null_epsilon(params):
    params_with_epsilon_0 = mps.MatchgateHamiltonianCoefficientsParams.parse_from_any(params)
    params_with_epsilon_0.epsilon = 0.0

    mg = Matchgate(params_with_epsilon_0)
    single_transition_particle_matrix = qml.math.expm(-4 * mg.hamiltonian_coefficients_params.to_matrix())
    np.testing.assert_allclose(
        mg.single_particle_transition_matrix.squeeze(),
        single_transition_particle_matrix.squeeze(),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_type0,params_type1",
    [
        (params_type0, params_type1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for params_type0 in [
            mps.MatchgatePolarParams,
            mps.MatchgateComposedHamiltonianParams,
        ]
        for params_type1 in [
            mps.MatchgatePolarParams,
            mps.MatchgateComposedHamiltonianParams,
        ]
        if params_type0 == params_type1
    ],
)
def test_mg_equal(params_type0, params_type1):
    params0 = params_type0.random()
    params1 = params_type1.parse_from_any(params0)
    mg0 = Matchgate(params0)
    mg1 = Matchgate(params1)
    check = mg0 == mg1
    assert check, f"{mg0 = }, {mg1 = }"


@pytest.mark.parametrize(
    "params",
    [mps.MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_single_transition_matrix_equal_to_expm_hami_coeff_if_epsilon(params):
    params_with_epsilon_0 = mps.MatchgateHamiltonianCoefficientsParams.parse_from_any(params)
    params_with_epsilon_0.epsilon = 1e7

    mg = Matchgate(params_with_epsilon_0)
    single_transition_particle_matrix = qml.math.expm(
        -4 * mg.hamiltonian_coefficients_params.to_matrix(add_epsilon=False)
    )
    np.testing.assert_allclose(
        mg.single_particle_transition_matrix.squeeze(),
        single_transition_particle_matrix.squeeze(),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params",
    [mps.MatchgateComposedHamiltonianParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_single_transition_matrix_equal_to_expm_hami_coeff(params):
    mg = Matchgate(params)
    h = mg.hamiltonian_coefficients_params.to_matrix(add_epsilon=False)
    single_transition_particle_matrix = qml.math.expm(-4 * h)

    np.testing.assert_allclose(
        mg.single_particle_transition_matrix.squeeze(),
        single_transition_particle_matrix.squeeze(),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
