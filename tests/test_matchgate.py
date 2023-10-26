import numpy as np
import pytest
from msim import Matchgate


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
        for _ in range(100)
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
    assert np.allclose(mg.data, input_matrix, rtol=1.e-4, atol=1.e-5), (f"The output matrix is not the correct one. "
                                                                        f"Got \n{mg.data} instead of \n{input_matrix}")


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
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
    ]
)
def test_matchgate_hamiltonian_coefficient(input_matrix, target_coefficients):
    mg = Matchgate.from_matrix(input_matrix)
    coeffs_vector = mg.hamiltonian_coeffs
    coeff_check = np.allclose(coeffs_vector, target_coefficients, rtol=1.e-4, atol=1.e-5)
    out_matchgate = Matchgate.from_hamiltonian_coeffs(coeffs_vector)
    mg_check = np.allclose(out_matchgate.data, mg.data, rtol=1.e-2, atol=1.e-3)
    assert coeff_check, (f"The output vector is not the correct one. "
                         f"Got {coeffs_vector} instead of {target_coefficients}")
    assert mg_check, f"The output matchgate is not the correct one. Got \n{out_matchgate.data} instead of \n{mg.data}"


