import numpy as np
import pytest
from msim import Matchgate, mps, utils
from .configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.fixture
def matchgate_rn_init(*args, **kwargs) -> Matchgate:
    return Matchgate.random()


def test_matchgate_m_m_dagger_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_m_m_dagger_constraint()


def test_matchgate_m_dagger_m_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_m_dagger_m_constraint()


def test_matchgate_det_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_det_constraint()


@pytest.mark.parametrize(
    "params",
    [
        tuple(np.random.rand(6))
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_matchgate_constructor_with_default_theta4(params):
    m = Matchgate(params)
    exp4 = np.exp(1j * np.mod(m.polar_params.theta4, 2 * np.pi))
    exp2 = np.exp(1j * np.mod(m.polar_params.theta2, 2 * np.pi))
    assert np.isclose(exp2 * exp4, 1.0+0j), f"{exp2 * exp4 = }, expected 1.0+0j"


@pytest.mark.parametrize(
    "input_matrix",
    [
        np.array([
            [1.0, 0, 0, 0.0],
            [0, 1.0, 0.0, 0],
            [0, 0.0, 1.0, 0],
            [0.0, 0, 0, 1.0],
        ]),
        np.array([
            [1.0, 0, 0, 0.0],
            [0, 0.0, 1.0, 0],
            [0, -1.0, 0.0, 0],
            [0.0, 0, 0, 1.0],
        ]),
    ]
)
def test_matchgate_constructor_from_matrix(input_matrix):
    mg = Matchgate.from_matrix(input_matrix)
    assert np.allclose(mg.gate_data, input_matrix, rtol=1.e-4, atol=1.e-5), (
        f"The output matrix is not the correct one. Got \n{mg.gate_data} instead of \n{input_matrix}"
    )


@pytest.mark.parametrize(
    "input_matrix,target_coefficients",
    [
        (
                np.array([
                    [1.0, 0, 0, 0.0],
                    [0, 1.0, 0.0, 0],
                    [0, 0.0, 1.0, 0],
                    [0.0, 0, 0, 1.0]
                ]),
                np.zeros(mps.MatchgateHamiltonianCoefficientsParams.N_PARAMS)
        )
    ]
)
def test_matchgate_hamiltonian_coefficient(input_matrix, target_coefficients):
    mg = Matchgate.from_matrix(input_matrix)
    coeffs_vector = mg.hamiltonian_coefficients_params.to_numpy()
    coeff_check = np.allclose(coeffs_vector, target_coefficients, rtol=1.e-4, atol=1.e-5)
    assert coeff_check, (f"The output vector is not the correct one. "
                         f"Got {coeffs_vector} instead of {target_coefficients}")


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_random_polar_params_gives_matchgate(params):
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    assert matchgate.check_m_m_dagger_constraint(), f"m_m_dagger_constraint failed for random {type(params)}"
    assert matchgate.check_m_dagger_m_constraint(), f"m_dagger_m_constraint failed for random {type(params)}"
    assert matchgate.check_det_constraint(), f"det_constraint failed for random {type(params)}"
    

@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams.random()[:-1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_random_simple_polar_params_respect_constraint_in_hamiltonian_form(params):
    from scipy.linalg import expm
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    gate_det = np.linalg.det(matchgate.gate_data)
    hamiltonian_form_det = np.linalg.det(expm(1j * matchgate.hamiltonian_matrix))
    hamiltonian_trace = np.trace(matchgate.hamiltonian_matrix)
    exp_trace = np.exp(1j * hamiltonian_trace)
    assert np.isclose(hamiltonian_form_det, gate_det), f"{hamiltonian_form_det = }, {gate_det = }"
    assert np.isclose(gate_det, exp_trace), f"{gate_det = }, {exp_trace = }"
    

@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_random_polar_params_respect_constraint_in_hamiltonian_form(params):
    from scipy.linalg import expm
    matchgate = Matchgate(params, raise_errors_if_not_matchgate=False)
    gate_det = np.linalg.det(matchgate.gate_data)
    hamiltonian_form_det = np.linalg.det(expm(1j * matchgate.hamiltonian_matrix))
    hamiltonian_trace = np.trace(matchgate.hamiltonian_matrix)
    exp_trace = np.exp(1j * hamiltonian_trace)
    assert np.isclose(hamiltonian_form_det, gate_det), f"{hamiltonian_form_det = }, {gate_det = }"
    assert np.isclose(gate_det, exp_trace), f"{gate_det = }, {exp_trace = }"


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgateComposedHamiltonianParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
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
    ]
)
def test_action_matrix(params, expected):
    mg = Matchgate(params)
    action_matrix = mg.single_transition_particle_matrix
    check = np.allclose(action_matrix, expected)
    assert check, f"The action matrix is not the correct one. Got \n{action_matrix} instead of \n{expected}"


@pytest.mark.parametrize(
    "params,expected",
    [
        (
                mps.MatchgatePolarParams(r0=1, r1=1),
                np.array([
                    [0.25 * np.trace(utils.get_majorana(i, 2) @ utils.get_majorana(j, 2)) for j in range(4)]
                    for i in range(4)
                ])
        ),
        (
                mps.MatchgatePolarParams(r0=1, r1=1),
                np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ])
        )
    ]
)
def test_single_transition_matrix(params, expected):
    mg = Matchgate(params)
    check = np.allclose(mg.single_transition_particle_matrix, expected)
    assert check, (f"The single transition particle matrix is not the correct one. "
                   f"Got \n{mg.single_transition_particle_matrix} instead of \n{expected}")


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_transition_matrix_equal_to_expm_hami_coeff_if_null_epsilon(params):
    from scipy.linalg import expm

    params_with_epsilon_0 = mps.MatchgateHamiltonianCoefficientsParams.parse_from_any(params)
    params_with_epsilon_0.epsilon = 0.0

    mg = Matchgate(params_with_epsilon_0)

    single_transition_particle_matrix = expm(-4 * mg.hamiltonian_coefficients_params.to_matrix())

    check = np.allclose(mg.single_transition_particle_matrix, single_transition_particle_matrix)
    assert check, (f"The single transition particle matrix is not the correct one. "
                   f"Got \n{mg.single_transition_particle_matrix} instead of \n{single_transition_particle_matrix}")


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_transition_matrix_equal_to_expm_hami_coeff(params):
    from scipy.linalg import expm

    mg = Matchgate(params)
    h = (
            mg.hamiltonian_coefficients_params.to_matrix(add_epsilon=False)
            # +
            # mg.hamiltonian_coefficients_params.epsilon * np.eye(4)
    )
    single_transition_particle_matrix = expm(-4 * h) * expm(1j * mg.hamiltonian_coefficients_params.epsilon * np.eye(4))

    check = np.allclose(mg.single_transition_particle_matrix, single_transition_particle_matrix)
    assert check, (f"The single transition particle matrix is not the correct one. "
                   f"Got \n{mg.single_transition_particle_matrix} instead of \n{single_transition_particle_matrix}")
