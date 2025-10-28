import numpy as np
import pytest
import torch
from scipy.linalg import expm
from torch.autograd import gradcheck

from matchcake import matchgate_parameter_sets as mps
from matchcake import utils

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestUtils:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "input_vector,target_matrix",
        [
            (
                np.array([1]),
                np.array(
                    [
                        [0, 1],
                        [-1, 0],
                    ]
                ),
            ),
            (np.array([1, 2]), ValueError),
            (np.array([1, 2, 3]), np.array([[0, 1, 2], [-1, 0, 3], [-2, -3, 0]])),
            (np.array([1, 2, 3, 4]), ValueError),
            (
                np.array([1, 2, 3, 4, 5, 6]),
                np.array([[0, 1, 2, 3], [-1, 0, 4, 5], [-2, -4, 0, 6], [-3, -5, -6, 0]]),
            ),
        ],
    )
    def test_skew_antisymmetric_vector_to_matrix(self, input_vector, target_matrix):
        if isinstance(target_matrix, np.ndarray):
            out_matrix = utils.skew_antisymmetric_vector_to_matrix(input_vector)
            np.testing.assert_allclose(out_matrix, target_matrix)

        elif issubclass(target_matrix, BaseException):
            with pytest.raises(target_matrix):
                out_matrix = utils.skew_antisymmetric_vector_to_matrix(input_vector)


    @pytest.mark.parametrize(
        "state,hamming_weight",
        [
            #          0
            (np.array([1, 0]), 0),
            (np.array([0, 1]), 1),
            #          0     1
            (np.array([1, 0, 1, 0]), 0),
            (np.array([1, 0, 0, 1]), 1),
            (np.array([0, 1, 0, 1]), 2),
            #          0     1     2
            (np.array([1, 0, 1, 0, 1, 0]), 0),
            (np.array([1, 0, 1, 0, 0, 1]), 1),
            (np.array([1, 0, 0, 1, 0, 1]), 2),
            (np.array([0, 1, 0, 1, 0, 1]), 3),
            #          0     1     2     3
            (np.array([1, 0, 1, 0, 1, 0, 1, 0]), 0),
            (np.array([1, 0, 1, 0, 1, 0, 0, 1]), 1),
            (np.array([1, 0, 1, 0, 0, 1, 0, 1]), 2),
            (np.array([1, 0, 0, 1, 0, 1, 0, 1]), 3),
            (np.array([0, 1, 0, 1, 0, 1, 0, 1]), 4),
        ],
    )
    def test_get_hamming_weight(self, state, hamming_weight):
        out_hamming_weight = utils.get_hamming_weight(state)
        np.testing.assert_allclose(
            out_hamming_weight,
            hamming_weight,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )


    @pytest.mark.parametrize(
        "coeffs,hamiltonian",
        [
            (
                mps.MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
                -2j
                * np.array(
                    [
                        [2j, 0, 0, 2j],
                        [0, 0, -2, 0],
                        [0, 2, 0, 0],
                        [2j, 0, 0, -2j],
                    ]
                ),
            ),
        ],
    )
    def test_get_non_interacting_fermionic_hamiltonian_from_coeffs(self, coeffs, hamiltonian):
        out_hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(coeffs.to_matrix())
        np.testing.assert_allclose(
            out_hamiltonian.squeeze(),
            hamiltonian.squeeze(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("coefficients_size", [2**2, ])
    def test_decompose_matrix_into_majoranas(self, coefficients_size):
        coefficients = np.random.rand(coefficients_size)
        matrix = np.zeros((coefficients.size, coefficients.size), dtype=complex)
        n = int(np.log2(coefficients.size))
        for i in range(coefficients.size):
            matrix += coefficients[i] * utils.get_majorana(i, n)

        out_coefficients = utils.decompose_matrix_into_majoranas(matrix)
        np.testing.assert_allclose(
            out_coefficients,
            coefficients,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("size", [4, 6])
    def test_make_transition_matrix_from_action_matrix(self, size):
        matrix = np.random.rand(size, size)
        t_matrix = utils.make_transition_matrix_from_action_matrix(matrix)

        reconstructed_matrix = np.zeros_like(matrix)
        reconstructed_matrix[:, ::2] = 2 * np.real(t_matrix).T
        reconstructed_matrix[:, 1::2] = 2 * np.imag(t_matrix).T
        np.testing.assert_allclose(
            reconstructed_matrix,
            matrix,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "batch_size, size",
        [
            (1, 3),
            (3, 1),
            (3, 4),
        ],
    )
    def test_make_transition_matrix_from_action_matrix_gradients(self, batch_size, size):
        matrix = expm(np.random.randn(batch_size, 2 * size, 2 * size))
        params = torch.from_numpy(matrix).requires_grad_()
        assert gradcheck(
            utils.make_transition_matrix_from_action_matrix,
            (params,),
            eps=1e-3,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "vector_length", list(range(1, 10)),
    )
    def test_binary_string_to_vector(self, vector_length):
        binary_vector = np.random.randint(0, 2, size=vector_length)
        binary_string = "".join(str(x) for x in binary_vector)
        binary_vector = np.asarray(binary_vector)
        out_binary_vector = utils.binary_string_to_vector(binary_string)
        np.testing.assert_allclose(
            out_binary_vector,
            binary_vector,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )


    @pytest.mark.parametrize(
        "inputs,out_state",
        [
            ((0, 2), "00"),
            ((1, 2), "01"),
            ((2, 2), "10"),
            ((3, 2), "11"),
            ((0, 3), "000"),
            ((1, 3), "001"),
            ((2, 3), "010"),
            ((3, 3), "011"),
            ((4, 3), "100"),
            ((5, 3), "101"),
            ((6, 3), "110"),
            ((7, 3), "111"),
            (np.array([1, 0]), "0"),
            (np.array([0, 1]), "1"),
            (np.array([1, 0, 0, 0]), "00"),
            (np.array([0, 1, 0, 0]), "01"),
            (np.array([0, 0, 1, 0]), "10"),
            (np.array([0, 0, 0, 1]), "11"),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0]), "000"),
            (np.array([0, 1, 0, 0, 0, 0, 0, 0]), "001"),
            (np.array([0, 0, 1, 0, 0, 0, 0, 0]), "010"),
            (np.array([0, 0, 0, 1, 0, 0, 0, 0]), "011"),
            (np.array([0, 0, 0, 0, 1, 0, 0, 0]), "100"),
            (np.array([0, 0, 0, 0, 0, 1, 0, 0]), "101"),
            (np.array([0, 0, 0, 0, 0, 0, 1, 0]), "110"),
            (np.array([0, 0, 0, 0, 0, 0, 0, 1]), "111"),
        ],
    )
    def test_state_to_binary_state(self, inputs, out_state):
        if isinstance(inputs, tuple):
            binary_state = utils.state_to_binary_string(*inputs)
        else:
            binary_state = utils.state_to_binary_string(inputs)
        assert binary_state == out_state, f"{binary_state} != {out_state}"
