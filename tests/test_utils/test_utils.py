import numpy as np
import pennylane as qml
import pytest
import torch
from scipy.linalg import expm
from torch.autograd import gradcheck

from matchcake import utils
from matchcake.utils import MajoranaGetter, get_majorana
from matchcake.utils.operators import recursive_2in_operator

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
                dict(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
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
        coeffs_matrix = np.array(
            [
                [0, coeffs["h0"], coeffs["h1"], coeffs["h2"]],
                [-coeffs["h0"], 0, coeffs["h3"], coeffs["h4"]],
                [-coeffs["h1"], -coeffs["h3"], 0, coeffs["h4"]],
                [-coeffs["h2"], -coeffs["h4"], -coeffs["h5"], 0],
            ]
        )
        out_hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(coeffs_matrix)
        np.testing.assert_allclose(
            out_hamiltonian.squeeze(),
            hamiltonian.squeeze(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "coefficients_size",
        [
            2**2,
        ],
    )
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
            (1, 4),
            (3, 2),
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
        "vector_length",
        list(range(1, 10)),
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

    def test_binary_state_to_state_with_list(self):
        result = utils.binary_state_to_state([0, 1])
        np.testing.assert_allclose(result, np.array([0, 1, 0, 0]))

    def test_state_to_binary_state_from_array(self):
        result = utils.state_to_binary_state(np.array([0, 0, 1, 0]))
        np.testing.assert_array_equal(result, np.array([1, 0]))

    def test_binary_string_to_state_number(self):
        assert utils.binary_string_to_state_number("10") == 2
        assert utils.binary_string_to_state_number("11") == 3
        assert utils.binary_string_to_state_number("01") == 1

    def test_get_non_interacting_fermionic_hamiltonian_3d(self):
        coeffs = np.random.randn(1, 4, 4)
        h = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(coeffs)
        assert h.shape == (1, 4, 4)

    def test_get_non_interacting_fermionic_hamiltonian_wrong_ndim(self):
        coeffs = np.random.randn(2, 4, 4, 4)
        with pytest.raises(ValueError):
            utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(coeffs)

    def test_decompose_matrix_into_majoranas_with_getter(self):
        n = 2
        coefficients = np.random.rand(2**n)
        matrix = sum(coefficients[i] * get_majorana(i, n) for i in range(2**n))
        getter = MajoranaGetter(n=n)
        out = utils.decompose_matrix_into_majoranas(matrix, majorana_getter=getter)
        np.testing.assert_allclose(out, coefficients, atol=ATOL_MATRIX_COMPARISON)

    def test_decompose_state_into_majorana_indexes(self):
        state = np.array([0, 1, 0, 0])
        indexes = utils.decompose_state_into_majorana_indexes(state)
        np.testing.assert_array_equal(indexes, np.array([2]))

    def test_decompose_state_into_majorana_indexes_int(self):
        indexes = utils.decompose_state_into_majorana_indexes(1, n=2)
        np.testing.assert_array_equal(indexes, np.array([2]))

    def test_get_unitary_from_hermitian_matrix(self):
        h = np.diag([1.0, 2.0, 3.0, 4.0])
        u = utils.get_unitary_from_hermitian_matrix(h)
        assert u.shape == (4, 4)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(4), atol=ATOL_MATRIX_COMPARISON)

    def test_load_backend_lib_string(self):
        lib = utils.load_backend_lib("numpy")
        import numpy as _np

        assert lib is _np

    def test_camel_case_to_spaced_camel_case(self):
        result = utils.camel_case_to_spaced_camel_case("CamelCaseString")
        assert result == "Camel Case String"

    def test_camel_case_to_spaced_camel_case_no_caps(self):
        result = utils.camel_case_to_spaced_camel_case("lowercase")
        assert result == "lowercase"

    def test_get_probabilities_from_state_int_wires(self):
        state = np.array([1.0, 0.0, 0.0, 0.0])
        probs = utils.get_probabilities_from_state(state, wires=0)
        np.testing.assert_allclose(probs, np.array([1.0, 0.0]), atol=ATOL_SCALAR_COMPARISON)

    def test_get_all_subclasses_include_base(self):
        class _Base:
            pass

        class _Sub(_Base):
            pass

        result = utils.get_all_subclasses(_Base, include_base_cls=True)
        assert _Base in result
        assert _Sub in result

    def test_get_eigvals_on_z_basis_raise_on_failure(self):
        class _BadOp:
            wires = qml.wires.Wires([0, 1])

            def matrix(self):
                raise RuntimeError("no matrix")

        with pytest.raises(RuntimeError):
            utils.get_eigvals_on_z_basis(_BadOp(), raise_on_failure=True)

    def test_recursive_2in_operator_non_recursive(self):
        result = recursive_2in_operator(np.add, [1, 2, 3, 4], recursive=False)
        assert result == 10

    def test_recursive_2in_operator_empty_raises(self):
        with pytest.raises(ValueError):
            recursive_2in_operator(np.add, [])
