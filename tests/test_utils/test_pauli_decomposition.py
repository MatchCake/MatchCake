import itertools
from typing import Callable, Sequence
from unittest import mock

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane import numpy as pnp
from pennylane.ops import LinearCombination
from pennylane.ops.op_math import Adjoint, ChangeOpBasis, Pow, Prod, SProd, Sum
from pennylane.pauli import PauliWord

import matchcake.utils.pauli_decomposition as pauli_decomposition_module
from matchcake.utils.pauli_decomposition import (
    has_live_terms,
    live_pauli_sentence,
    live_pauli_terms,
    pauli_word_product,
    stacked_coefficients,
)

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)

WIRES = [0, 1, 2]

# Single-qubit Pauli matrices, written out so the oracle shares nothing with PennyLane's algebra.
PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
PAULI_NAMES = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}

# Separates arithmetic done in double precision (error ~1e-17 here) from arithmetic done in float32
# before the final cast (error ~7e-10 for 0.5 * 0.1).
DOUBLE_PRECISION_ATOL = 1e-12

# Nesting depth past CPython's default recursion limit of 1000, as a qml.sum(H, term) loop over a
# Hamiltonian of that many terms builds.
DEEP_NESTING = 1500


class TestPauliDecomposition:
    @staticmethod
    def scalar_value(value: object) -> complex:
        """A python, numpy or torch scalar, including a one-element array, as a python ``complex``.

        :param value: The scalar.
        :return: Its value.
        :rtype: complex
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return complex(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def word_matrix(word: PauliWord, wires: Sequence = WIRES) -> np.ndarray:
        """Dense matrix of a Pauli word from :data:`PAULI_MATRICES`, big-endian over ``wires``.

        :param word: The Pauli word.
        :param wires: Wire labels in qubit order.
        :return: Dense ``(2**n, 2**n)`` matrix.
        :rtype: np.ndarray
        """
        matrix = np.ones((1, 1), dtype=complex)
        for wire in wires:
            matrix = np.kron(matrix, PAULI_MATRICES[word.get(wire, "I")])
        return matrix

    @classmethod
    def dense_operator(cls, operator: qml.operation.Operator, wires: Sequence = WIRES) -> np.ndarray:
        """Dense matrix of an observable, rebuilt by walking its tree with coefficients read live.

        ``qml.matrix`` evaluates sums and products through the cached ``pauli_rep``, so it would
        agree with a stale reading; this oracle does not touch it. A non-Pauli leaf (a gate such as
        ``S`` or ``SWAP``) is evaluated through its own matrix, which is independent of the
        decomposition.

        :param operator: The observable or a node of it.
        :param wires: Wire labels in qubit order.
        :return: Dense ``(2**n, 2**n)`` matrix.
        :rtype: np.ndarray
        """
        size = 2 ** len(wires)
        zero = np.zeros((size, size), dtype=complex)
        if isinstance(operator, LinearCombination):
            terms = zip(operator.coeffs, operator.ops)
            return sum((cls.scalar_value(weight) * cls.dense_operator(op, wires) for weight, op in terms), zero)
        if isinstance(operator, Sum):
            return sum((cls.dense_operator(op, wires) for op in operator.operands), zero)
        if isinstance(operator, SProd):
            return cls.scalar_value(operator.scalar) * cls.dense_operator(operator.base, wires)
        if isinstance(operator, (Prod, ChangeOpBasis)):
            matrix = np.eye(size, dtype=complex)
            for op in operator.operands:
                matrix = matrix @ cls.dense_operator(op, wires)
            return matrix
        if isinstance(operator, Pow):
            return np.linalg.matrix_power(cls.dense_operator(operator.base, wires), int(operator.z))
        if isinstance(operator, Adjoint):
            return cls.dense_operator(operator.base, wires).conj().T
        if operator.name == "Identity":
            return np.eye(size, dtype=complex)
        if operator.name in PAULI_NAMES:
            return cls.word_matrix(PauliWord({operator.wires[0]: PAULI_NAMES[operator.name]}), wires)
        return qml.matrix(operator, wire_order=wires)

    @classmethod
    def reconstruct(cls, sentence: dict, wires: Sequence = WIRES) -> np.ndarray:
        """Dense ``sum_k coefficient_k P_k`` of a decomposition.

        :param sentence: ``{Pauli word: coefficient}``.
        :param wires: Wire labels in qubit order.
        :return: Dense ``(2**n, 2**n)`` matrix.
        :rtype: np.ndarray
        """
        size = 2 ** len(wires)
        return sum(
            (cls.scalar_value(value) * cls.word_matrix(word, wires) for word, value in sentence.items()),
            np.zeros((size, size), dtype=complex),
        )

    @pytest.mark.parametrize("left, right", list(itertools.product("IXYZ", repeat=2)))
    def test_every_single_qubit_product_matches_the_matrix_product(self, left: str, right: str) -> None:
        left_word = PauliWord({0: left}) if left != "I" else PauliWord({})
        right_word = PauliWord({0: right}) if right != "I" else PauliWord({})
        word, phase = pauli_word_product(left_word, right_word)
        np.testing.assert_allclose(
            phase * self.word_matrix(word, [0]),
            PAULI_MATRICES[left] @ PAULI_MATRICES[right],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_multi_qubit_product_combines_the_wires_independently(self) -> None:
        left = PauliWord({0: "X", 1: "Y", 2: "Z"})
        right = PauliWord({0: "Y", 1: "Y", "a": "X"})
        word, phase = pauli_word_product(left, right)
        wires = [0, 1, 2, "a"]
        np.testing.assert_allclose(
            phase * self.word_matrix(word, wires),
            self.word_matrix(left, wires) @ self.word_matrix(right, wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "observable",
        [
            qml.Hamiltonian([0.5], [-qml.Z(0) @ qml.Z(1)]),
            qml.Hamiltonian([1.0], [2.0 * qml.X(0)]),
            qml.X(0) @ qml.Y(0),
            qml.Hamiltonian([1.0, 0.5], [qml.X(0), qml.Z(1)]) - qml.Hamiltonian([0.3, 2.0], [qml.Y(0), qml.X(1)]),
            qml.Hamiltonian([1.0, 0.5], [qml.X(0), qml.Z(1)]) ** 2,
            qml.adjoint(qml.s_prod(1j, qml.X(0) @ qml.Y(1) @ qml.Z(2))),
            qml.pow(2.0 * qml.X(0), -1),
            qml.pow(qml.X(0) @ qml.Y(1), -3),
            qml.Z(0) + qml.pow(qml.X(1), 0),
            qml.sum(qml.Hamiltonian([], []), 0.5 * qml.Z(0)),
            qml.change_op_basis(qml.S(0), qml.X(0)),
            qml.change_op_basis(qml.X(0), qml.Hamiltonian([1.0, 0.5], [qml.Z(0), qml.Z(1)])),
            qml.SWAP(wires=[0, 1]),
            (lambda shared: qml.sum(shared, shared, qml.prod(shared, shared)))(qml.Z(0) @ qml.X(1)),
            qml.dot([0.5, -0.3], [qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Y(2)]),
            qml.Hamiltonian([0.5, 2.0], [qml.Identity(0), -qml.Z(0)]),
            qml.prod(
                qml.Hamiltonian([1.0], [qml.X(0) @ qml.Y(1)]),
                qml.Hamiltonian([1.0], [qml.Y(0) @ qml.Z(1)]),
            ),
        ],
    )
    def test_an_observable_is_reconstructed_by_its_decomposition(self, observable: qml.operation.Operator) -> None:
        np.testing.assert_allclose(
            self.reconstruct(live_pauli_sentence(observable)),
            self.dense_operator(observable),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_the_terms_reproduce_the_sentence_with_fresh_words(self) -> None:
        observable = 0.5 * (qml.Z(0) @ qml.Z(1)) - 0.25 * qml.X(2)
        coefficients, operators = live_pauli_terms(observable)
        sentence = live_pauli_sentence(observable)
        assert [op.pauli_rep for op in operators] == [qml.pauli.PauliSentence({word: 1.0}) for word in sentence]
        np.testing.assert_allclose(
            [self.scalar_value(value) for value in coefficients],
            [self.scalar_value(value) for value in sentence.values()],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "observable, expected",
        [
            (qml.Z(0), True),
            (qml.Identity(), True),
            (qml.X(0) @ qml.Y(1) @ qml.Z(2), True),
            (qml.Hamiltonian([0.5, 0.3], [qml.Z(0) @ qml.Z(1), qml.Identity(0)]), True),
            (qml.X(0) @ qml.Y(0), False),
            (2.0 * qml.Z(0), False),
            (qml.Z(0) + qml.X(1), False),
            (qml.Hamiltonian([0.5], [-qml.Z(0)]), False),
            (qml.Hamiltonian([0.5], [qml.Hamiltonian([1.0], [qml.Z(0)])]), False),
            (qml.SWAP(wires=[0, 1]), False),
            (qml.S(0) @ qml.Z(1), False),
            (qml.Hamiltonian([0.5, 0.3], [qml.Z(0), qml.Z(0)], grouping_type="qwc"), True),
        ],
    )
    def test_only_plain_words_and_hamiltonians_of_them_count_as_live(
        self, observable: qml.operation.Operator, expected: bool
    ) -> None:
        """A word on a repeated wire carries a phase of its own, and so does a scalar product. Grouping
        does not change the answer: it must not flip for one object, which a cached decomposition keys on."""
        assert has_live_terms(observable) is expected

    def test_a_hamiltonian_of_plain_words_is_returned_as_it_is(self) -> None:
        """The cheap path: its own coefficient array and operators, terms neither merged nor rebuilt."""
        weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        operators = [qml.Z(0) @ qml.Z(1), qml.X(2), qml.Z(0) @ qml.Z(1)]
        coefficients, returned = live_pauli_terms(qml.Hamiltonian(weights, operators))
        assert all(returned_op is op for returned_op, op in zip(returned, operators))
        np.testing.assert_allclose(
            [float(value) for value in coefficients],
            [0.5, 0.3, 0.2],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_plain_word_is_one_unit_term(self) -> None:
        observable = qml.X(0) @ qml.Z(2)
        coefficients, operators = live_pauli_terms(observable)
        assert operators == [observable]
        np.testing.assert_allclose(coefficients, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("interface", ["torch", "numpy", "autograd"])
    @pytest.mark.parametrize(
        "wrap",
        [
            lambda inner: qml.Hamiltonian([1.0], [qml.Z(2)]) - inner,
            lambda inner: qml.sum(qml.Z(2), inner),
            lambda inner: qml.adjoint(inner),
            lambda inner: qml.pow(inner, 1),
            lambda inner: qml.Hamiltonian([0.5], [inner]),
            lambda inner: qml.prod(inner, qml.X(2)),
        ],
    )
    def test_an_in_place_update_is_seen_at_any_depth(self, wrap, interface: str) -> None:
        """A Hamiltonian's cached ``pauli_rep``, and with numpy weights its ``operands``, keep the old
        values; only ``coeffs`` follows the update, and only the walk reads it at every level."""
        if interface == "torch":
            weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        else:
            weights = np.array([0.5, 0.3]) if interface == "numpy" else pnp.array([0.5, 0.3])
        observable = wrap(qml.Hamiltonian(weights, [qml.Z(0), qml.X(0) @ qml.X(1)]))
        live_pauli_sentence(observable)
        with torch.no_grad():
            weights[0] = 2.0
        np.testing.assert_allclose(
            self.reconstruct(live_pauli_sentence(observable)),
            self.dense_operator(observable),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("interface", ["torch", "autograd"])
    def test_reuse_in_a_sum_does_not_corrupt_a_wrapped_observable(self, interface: str) -> None:
        """``A + A`` adds into the shared coefficient tensors of ``A``'s cached ``pauli_rep``."""
        weights = torch.tensor([1.0, 0.25], dtype=torch.float64) if interface == "torch" else pnp.array([1.0, 0.25])
        inner = qml.Hamiltonian(weights, [qml.Z(0), qml.X(0) @ qml.Y(2)])
        observable = qml.Hamiltonian([0.5], [qml.adjoint(inner)])
        _ = inner + inner
        np.testing.assert_allclose(
            self.reconstruct(live_pauli_sentence(observable)),
            self.dense_operator(observable),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_decomposing_neither_mutates_the_observable_nor_drifts(self) -> None:
        """``Sum.terms()`` falls back to simplification for a shape-``(1,)`` scalar and accumulates
        into the caller's tensor in place; the walk performs no in-place operation at all."""
        weight = torch.tensor([0.5], dtype=torch.float64)
        observable = weight * qml.Z(0) + 2.0 * qml.Z(0) + qml.pow(qml.X(1), 0)
        for _ in range(3):
            np.testing.assert_allclose(
                self.reconstruct(live_pauli_sentence(observable)),
                2.5 * self.dense_operator(qml.Z(0)) + np.eye(8),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
        np.testing.assert_allclose(weight.numpy(), [0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_the_coefficients_stay_differentiable(self) -> None:
        """The walk keeps coefficients in their own type, so a gradient flows back to every weight."""
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)

        def total(flat: torch.Tensor) -> torch.Tensor:
            observable = qml.Hamiltonian([1.0], [flat[0] * (qml.Z(0) @ qml.Z(1)) - flat[1] * qml.adjoint(qml.X(2))])
            sentence = live_pauli_sentence(observable)
            return sum(torch.as_tensor(value) * (index + 1) for index, value in enumerate(sentence.values()))

        assert torch.autograd.gradcheck(total, (weights,))

    def test_a_real_coefficient_on_commuting_factors_stays_real(self) -> None:
        """Only a product on a shared wire introduces a phase; real weights on distinct wires stay real."""
        weight = torch.tensor(0.5, dtype=torch.float64)
        sentence = live_pauli_sentence(qml.Hamiltonian([weight], [qml.X(0) @ qml.Y(1)]) + weight * qml.Z(2))
        assert all(not torch.is_complex(torch.as_tensor(value)) for value in sentence.values())

    def test_the_words_depend_on_the_structure_and_not_on_the_values(self) -> None:
        """A decomposition cached for an observable must still fit it after its weights change, so a
        word whose coefficient becomes zero is kept rather than dropped."""
        weights = np.array([0.5, -0.5])
        observable = qml.Hamiltonian(weights, [qml.Z(0), qml.Z(0)]) + qml.X(1)
        before = list(live_pauli_sentence(observable))
        weights[1] = 0.25
        assert list(live_pauli_sentence(observable)) == before

    @pytest.mark.parametrize("chain", ["sum", "adjoint", "prod"])
    def test_a_tree_nested_past_the_recursion_limit_is_expanded(self, chain: str) -> None:
        observable = qml.Z(0)
        for _ in range(DEEP_NESTING):
            if chain == "sum":
                observable = qml.sum(observable, qml.Z(0))
            elif chain == "adjoint":
                observable = qml.adjoint(observable)
            else:
                observable = qml.prod(observable, qml.Z(0))
        sentence = live_pauli_sentence(observable)
        expected = {"sum": {PauliWord({0: "Z"}): DEEP_NESTING + 1}, "adjoint": {PauliWord({0: "Z"}): 1}}
        # An odd number of Z factors in total, since DEEP_NESTING is even.
        expected["prod"] = {PauliWord({0: "Z"}): 1}
        assert list(sentence) == list(expected[chain])
        np.testing.assert_allclose(
            [self.scalar_value(value) for value in sentence.values()],
            list(expected[chain].values()),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "observable, match",
        [
            (qml.Hermitian(np.diag([1.0, -1.0]), wires=0), "no Pauli representation"),
            (qml.exp(qml.X(0), 0.5j), "no Pauli representation"),
            (qml.prod(qml.RX(0.0, wires=0), qml.X(0)), "no Pauli representation"),
            (qml.pow(qml.X(0), 0.5), "integer powers"),
            (qml.pow(qml.X(0), 1 + 1j), "integer powers"),
            (qml.pow(qml.X(0) + qml.Z(1), -1), "invert a single Pauli word"),
            (qml.pow(qml.X(0) + qml.Y(0) - qml.Y(0), -1), "invert a single Pauli word"),
            (qml.pow(qml.sum(qml.X(0), qml.pow(qml.Hamiltonian([], []), 3)), 0.5), "integer powers"),
            (qml.pow(qml.X(0) + qml.Z(1), torch.tensor(2), lazy=True), "plain number as exponent"),
        ],
    )
    def test_a_node_without_a_live_pauli_reading_is_refused(
        self, observable: qml.operation.Operator, match: str
    ) -> None:
        """The last case cannot even be printed by PennyLane, which must not replace the error."""
        with pytest.raises(TypeError, match=match):
            live_pauli_sentence(observable)

    def test_a_zero_power_is_the_identity_without_reading_its_base(self) -> None:
        sentence = live_pauli_sentence(qml.Z(0) + qml.pow(qml.Hadamard(1), 0))
        np.testing.assert_allclose(
            self.reconstruct(sentence),
            self.dense_operator(qml.Z(0)) + np.eye(8),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "observable, dtype",
        [
            (0.5 * qml.Z(0) + 0.25 * qml.X(1), np.float64),
            (qml.X(0) @ qml.Y(0) + qml.Z(1), np.complex128),
            (qml.Hamiltonian([0.5, 0.1], [qml.X(0) @ qml.Y(0) @ qml.X(0) @ qml.Y(0), qml.Z(1)]), np.float64),
        ],
    )
    def test_python_coefficients_come_back_as_one_double_precision_array(self, observable, dtype) -> None:
        """Real when every imaginary part is exactly zero, even after intermediate ``i`` phases, which
        is what keeps a Hermitian observable's expectation real and differentiable downstream."""
        coefficients, _ = live_pauli_terms(observable)
        assert isinstance(coefficients, np.ndarray)
        assert coefficients.dtype == dtype

    def test_torch_coefficients_mixed_with_constants_come_back_as_one_tensor(self) -> None:
        """PennyLane's ``Hamiltonian`` cannot take a list mixing a trainable tensor with a python
        float, and a python float multiplied into a torch result is rounded to float32."""
        weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        coefficients, _ = live_pauli_terms(0.5 * qml.Z(0) + weight * qml.Z(1))
        assert isinstance(coefficients, torch.Tensor)
        assert coefficients.dtype == torch.float64
        coefficients.sum().backward()
        np.testing.assert_allclose(float(weight.grad), 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_one_element_coefficient_is_read_as_a_scalar(self) -> None:
        """A shape-``(1,)`` weight next to scalar ones would otherwise give ``(n, 1)`` coefficients,
        which the strategies' einsums reject and which break circuit broadcasting."""
        weight = torch.tensor([0.5], dtype=torch.float64)
        coefficients, _ = live_pauli_terms(weight * qml.Z(0) + 0.25 * qml.Z(1))
        assert tuple(coefficients.shape) == (2,)

    def test_an_identity_term_keeps_the_wires_of_the_observable(self) -> None:
        """A strategy gathering probabilities over a term's wires would otherwise find none."""
        _, operators = live_pauli_terms(0.7 * qml.Identity(0))
        assert operators[0].wires.tolist() == [0]

    def test_a_part_without_a_pauli_representation_can_be_kept_whole(self) -> None:
        projector = qml.Projector([1], wires=[0])
        weight = torch.tensor(0.5, dtype=torch.float64)
        observable = projector + weight * qml.Z(1) + 0.25 * qml.Z(1)
        with pytest.raises(TypeError, match="no Pauli representation"):
            live_pauli_terms(observable)
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert operators[0] is projector
        np.testing.assert_allclose(
            coefficients.numpy(), [1.0, 0.75], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_product_with_a_part_kept_whole_is_one_term(self) -> None:
        """A non-Pauli factor cannot be multiplied through, so the whole product is one term."""
        product = qml.prod(qml.Hermitian(np.diag([1.0, -1.0]), wires=0), qml.Z(1))
        coefficients, operators = live_pauli_terms(0.5 * product + qml.X(2), keep_non_pauli=True)
        assert operators[0].hash == product.hash
        np.testing.assert_allclose(coefficients, [0.5, 1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_an_invalid_power_can_be_kept_as_a_term(self) -> None:
        """It is kept as PennyLane simplifies it, as ``Sum.terms()`` would give it."""
        power = qml.pow(qml.Hermitian(np.diag([1.0, -1.0]), wires=0), 0.5)
        _, operators = live_pauli_terms(power + qml.Z(1), keep_non_pauli=True)
        assert operators[0].hash == power.simplify().hash

    def test_an_invalid_power_simplified_to_a_gate_is_read_as_the_gate(self) -> None:
        """``X**0.5`` simplifies to ``SX``, which is read as its Pauli words, as ``SX`` written directly is."""
        coefficients, operators = live_pauli_terms(qml.pow(qml.X(0), 0.5) + qml.Z(1), keep_non_pauli=True)
        expected_coefficients, expected_operators = live_pauli_terms(qml.SX(0) + qml.Z(1))
        assert [operator.hash for operator in operators] == [operator.hash for operator in expected_operators]
        np.testing.assert_allclose(
            coefficients, expected_coefficients, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize(
        "observable, kind",
        [
            (qml.Z(0) + qml.Projector([0], wires=[1]) @ qml.Identity(2), qml.Projector),
            (qml.Z(0) + (2.0 * qml.Projector([0], wires=[1])) @ qml.Identity(2), qml.Projector),
            (
                qml.Z(0) + (qml.Projector([0], wires=[1]) + qml.Projector([0], wires=[2])) @ qml.Identity(0),
                qml.Projector,
            ),
            (qml.Z(0) + qml.Hadamard(1) @ qml.Hadamard(1), qml.Identity),
        ],
    )
    def test_a_part_kept_whole_is_simplified_as_sum_terms_simplifies_it(
        self, observable: qml.operation.Operator, kind: type
    ) -> None:
        """``P @ I`` becomes ``P`` and ``H @ H`` the identity, with sums and scalars taken apart."""
        _, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert any(isinstance(operator, kind) for operator in operators)
        assert not any(isinstance(operator, (qml.ops.Prod, qml.ops.SProd, qml.ops.Sum)) for operator in operators)

    @pytest.mark.parametrize(
        "base, kind",
        [
            (qml.Hamiltonian([2.0, 0.0], [qml.Z(1), qml.Projector([0], wires=[3])]), qml.Identity),
            (qml.Hamiltonian([2.0], [qml.Projector([0], wires=[1])]), qml.Projector),
        ],
    )
    def test_the_scalars_of_a_simplified_product_are_taken_out(self, base: qml.operation.Operator, kind: type) -> None:
        """``simplify`` leaves the square of ``2 A`` as ``(2 A) @ (2 A)``, which a strategy would read with
        coefficient one; expanding its factors gives ``4 A^2``."""
        coefficients, operators = live_pauli_terms(qml.Z(0) + qml.prod(base, base), keep_non_pauli=True)
        assert isinstance(operators[1], kind)
        np.testing.assert_allclose(coefficients, [1.0, 4.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_power_of_a_trainable_sum_is_reduced_without_raising(self) -> None:
        """A second ``simplify`` would add into the torch coefficients that the first one copied, which
        raises. ``(a P + b I)^2 = (a^2 + 2 a b) P + b^2 I``."""
        weights = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (-0.75, 0.5)]
        observable = qml.Z(0) + qml.pow(weights[0] * qml.Projector([1], wires=[1]) + weights[1] * qml.Identity(1), 2)
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        kinds = [qml.Z, qml.Projector, qml.Identity]
        assert len(operators) == len(kinds)
        assert all(isinstance(operator, kind) for operator, kind in zip(operators, kinds))
        np.testing.assert_allclose(
            coefficients.detach().numpy(),
            [1.0, -0.1875, 0.25],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "build, words, values",
        [
            # (X2 X3) @ Y2 @ Y3 = -Z2 Z3: simplify keeps the product, whose phase a strategy would drop.
            (
                lambda h: qml.prod(qml.prod(h(2.0, qml.X(2) @ qml.X(3)), qml.Y(2)), qml.Y(3)),
                [{2: "Z", 3: "Z"}],
                [-2.0],
            ),
            # A A^dagger with A = (1.25 X3 Y1) (2 Z1) = 2.5 i X3 X1 is 6.25 times the identity.
            (
                lambda h: qml.prod(
                    qml.prod(h(1.25, qml.X(3) @ qml.Y(1)), h(2.0, qml.Z(1))),
                    qml.adjoint(qml.prod(h(1.25, qml.X(3) @ qml.Y(1)), h(2.0, qml.Z(1)))),
                ),
                [{}],
                [6.25],
            ),
            # A A^dagger with A = (2 H0) (1.25 Z1): simplify leaves 6.25 H0 Z1 (Z1 H0), a product of products.
            (
                lambda h: qml.prod(
                    qml.prod(h(2.0, qml.Hadamard(0)), h(1.25, qml.Z(1))),
                    qml.adjoint(qml.prod(h(2.0, qml.Hadamard(0)), h(1.25, qml.Z(1)))),
                ),
                [{}],
                [6.25],
            ),
        ],
    )
    def test_a_product_left_unsettled_by_simplify_is_expanded_into_its_words(
        self, build: Callable, words: list, values: list
    ) -> None:
        def h(coefficient: float, operator: qml.operation.Operator) -> qml.operation.Operator:
            return qml.Hamiltonian([coefficient, 0.0], [operator, qml.Projector([0], wires=[0])])

        coefficients, operators = live_pauli_terms(build(h), keep_non_pauli=True)
        assert [list(operator.pauli_rep) for operator in operators] == [[PauliWord(word)] for word in words]
        np.testing.assert_allclose(
            np.real(coefficients), values, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_product_of_gates_is_simplified_until_it_settles(self) -> None:
        """``t t^dagger`` with ``t = CZ (-0.75 Y0 Y1 Y2 + 0 P) SWAP`` is ``0.5625``. One pass of ``simplify``
        leaves ``CZ @ Y0 @ Y1 @ Y0 @ Y1 @ CZ``, which a strategy refuses; the next ones give ``CZ @ CZ``,
        then the identity."""
        hamiltonian = qml.Hamiltonian([-0.75, 0.0], [qml.Y(0) @ qml.Y(1) @ qml.Y(2), qml.Projector([1], wires=[0])])
        part = qml.prod(qml.CZ([0, 1]), hamiltonian, qml.SWAP([0, 1]))
        coefficients, operators = live_pauli_terms(qml.prod(part, qml.adjoint(part)), keep_non_pauli=True)
        assert [list(operator.pauli_rep) for operator in operators] == [[PauliWord({})]]
        np.testing.assert_allclose(coefficients, [0.5625], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "factor",
        [
            lambda: qml.Hermitian(np.array([[1.0, 0.5 + 0.5j], [0.5 - 0.5j, -1.0]]), wires=0),
            lambda: qml.adjoint(qml.Projector([1], wires=[0])),
        ],
    )
    def test_a_product_that_simplify_returns_unchanged_is_kept_as_a_term(self, factor: Callable) -> None:
        """``simplify`` returns ``A @ A @ X`` as it is for these ``A``, which must end its expansion rather
        than start it again."""
        observable = qml.Z(0) + 2.0 * qml.prod(factor(), factor(), qml.X(1))
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert len(operators) == 2
        reconstructed = sum(
            self.scalar_value(coefficient) * self.dense_operator(operator, [0, 1])
            for coefficient, operator in zip(coefficients, operators)
        )
        np.testing.assert_allclose(
            reconstructed,
            self.dense_operator(observable, [0, 1]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("gate", [qml.SX, qml.S, qml.T])
    def test_a_gate_in_a_part_kept_whole_merges_with_its_words_read_elsewhere(self, gate: type) -> None:
        """In ``i (u + u^dagger)`` plus its adjoint, with ``u = G + 0 P``, each ``u^dagger`` is kept whole
        and simplifies to ``G^dagger``. Read as words, it cancels the words of ``G`` read in ``u``; kept
        as a gate, it would be measured on its own, real part only. The identity that ``simplify``
        puts on the gate's wire must merge with the one on the observable's wires."""
        part = qml.sum(gate(1), qml.Hamiltonian([0.0, 0.0], [qml.Z(2), qml.Projector([1], wires=[1])]))
        inner = 1j * (part + qml.adjoint(part))
        coefficients, operators = live_pauli_terms(qml.Z(0) + inner + qml.adjoint(inner), keep_non_pauli=True)
        assert [operator.hash for operator in operators] == [qml.Z(0).hash]
        np.testing.assert_allclose(coefficients, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_product_that_torch_cannot_simplify_is_read_through_its_terms(self) -> None:
        """``simplify`` merges ``X0 P1`` and ``P1 X0`` by adding in place into a copy of the trainable
        ``w0``, a leaf, which torch refuses; the product is multiplied out instead.
        ``(w0 X0 + w1 P1)(X0 + P1) = w0 I + (w0 + w1) X0 P1 + w1 P1``."""

        def build(requires_grad: bool) -> qml.operation.Operator:
            weights = torch.tensor([-1.5, 0.5], dtype=torch.float64, requires_grad=requires_grad)
            projector = qml.Projector([1], wires=[1])
            return qml.prod(qml.Hamiltonian(weights, [qml.X(0), projector]), qml.X(0) + projector)

        coefficients, operators = live_pauli_terms(build(True), keep_non_pauli=True)
        expected_coefficients, expected_operators = live_pauli_terms(build(False), keep_non_pauli=True)
        assert [operator.hash for operator in operators] == [operator.hash for operator in expected_operators]
        np.testing.assert_allclose(
            coefficients.detach().numpy(), [-1.5, -1.0, 0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            expected_coefficients, [-1.5, -1.0, 0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_product_that_torch_cannot_simplify_keeps_its_gradient(self) -> None:
        """The coefficients of ``(w0 X0 + w1 P1)(X0 + P1)``, multiplied out because torch refuses to
        simplify it, must be differentiable in the weights."""

        def coefficients(weights: torch.Tensor) -> torch.Tensor:
            projector = qml.Projector([1], wires=[1])
            observable = qml.prod(qml.Hamiltonian(weights, [qml.X(0), projector]), qml.X(0) + projector)
            return live_pauli_terms(observable, keep_non_pauli=True)[0]

        weights = torch.tensor([-1.5, 0.5], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(coefficients, (weights,))

    @pytest.mark.parametrize(
        "factor",
        [
            lambda: qml.pow(
                qml.Hamiltonian(
                    torch.tensor([1.0, 0.5], dtype=torch.float64, requires_grad=True),
                    [qml.Z(2), qml.Projector([1], wires=[2])],
                ),
                3,
            ),
            lambda: qml.pow(qml.Z(2) + 0.5 * qml.X(2), 0),
            lambda: qml.pow(2.0 * qml.X(2), -1),
            lambda: qml.change_op_basis(qml.Hadamard(2), qml.Z(2) + 0.5 * qml.X(2), qml.Hadamard(2)),
            lambda: qml.adjoint(qml.prod((1.0 + 0.5j) * qml.Hadamard(2), qml.Z(2))),
        ],
    )
    def test_a_node_in_a_product_torch_cannot_simplify_is_multiplied_out(self, factor: Callable) -> None:
        """A positive power is multiplied out as copies of its base, which torch could not simplify
        either, a basis change as the product of its operands, an adjoint with its coefficients
        conjugated, and a zero or negative power is left to ``simplify``."""
        weights = torch.tensor([-1.5, 0.5], dtype=torch.float64, requires_grad=True)
        projector = qml.Projector([1], wires=[1])
        observable = qml.prod(qml.Hamiltonian(weights, [qml.X(0), projector]), qml.X(0) + projector, factor())
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        reconstructed = sum(
            self.scalar_value(coefficient) * self.dense_operator(operator)
            for coefficient, operator in zip(coefficients, operators)
        )
        np.testing.assert_allclose(
            reconstructed, self.dense_operator(observable), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_a_term_is_multiplied_out_from_a_copy_the_failed_simplify_did_not_write_into(self) -> None:
        """``simplify`` adds into the copied coefficients before it raises, so multiplying out that copy
        would read them altered. ``i [L, H_2]`` vanishes for ``L`` off wire 2, so the observable is
        ``Z_0``."""
        wires = [0, 1, 2, 3]
        hadamard = qml.Hadamard(2)
        commuting = qml.Hamiltonian(
            torch.tensor([1j, 1.5j, 0.0], dtype=torch.complex128),
            [qml.X(1), qml.Y(0), qml.Projector([1], wires=[3])],
        )
        left = qml.Hamiltonian(
            torch.tensor([-0.75, 0.0], dtype=torch.float64), [qml.X(2), qml.Projector([1], wires=[3])]
        )
        right = qml.Hamiltonian(
            torch.tensor([1.5, -1.5, 0.0], dtype=torch.float64),
            [qml.Z(0) @ qml.Y(1) @ qml.X(3), qml.Z(0), qml.Projector([1], wires=[0])],
        )
        commutator = qml.s_prod(1j, commuting @ hadamard - hadamard @ commuting)
        observable = qml.Z(0) + qml.prod(left, commutator, qml.prod(hadamard, qml.Z(3), right))
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        reconstructed = sum(
            self.scalar_value(coefficient) * self.dense_operator(operator, wires)
            for coefficient, operator in zip(coefficients, operators)
        )
        np.testing.assert_allclose(
            reconstructed,
            self.dense_operator(qml.Z(0), wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_constant_zero_term_is_not_multiplied_out(self) -> None:
        """Each factor below has a zero-weighted projector, which ``simplify`` drops: the product of three
        multiplies out into one product, not eight. A trainable zero is kept for its gradient."""

        def product(requires_grad: bool) -> qml.operation.Operator:
            factors = [
                qml.Hamiltonian(
                    torch.tensor([0.5, 0.0], dtype=torch.float64, requires_grad=requires_grad),
                    [qml.X(0), qml.Projector([1], wires=[1])],
                )
                for _ in range(3)
            ]
            return qml.prod(*factors, qml.Z(2) + 0.0 * qml.Projector([1], wires=[2]))

        assert len(pauli_decomposition_module._distributed(product(False))) == 1
        assert len(pauli_decomposition_module._distributed(product(True))) == 8

    def test_a_product_that_fails_to_simplify_is_multiplied_out_exactly(self) -> None:
        """``t t^dagger`` with ``t = (Y1 + Z0 Z2)(0.25 Y3 + 2 Y0 + 0 P2 + 0.75 Y3 - 1.5 Y0 X3 + 0 H1)`` in
        torch weights fails to simplify. ``Prod.terms()`` would then add into copies that share
        storage with coefficients other products still read, and miss the matrix by ``1.5``."""
        wires = [0, 1, 2, 3]
        first = torch.tensor([0.25, 2.0, 0.0], dtype=torch.float64)
        second = torch.tensor([0.75, -1.5, 0.0], dtype=torch.float64)
        part = qml.prod(
            qml.Y(1) + qml.Z(0) @ qml.Z(2),
            qml.sum(
                qml.Hamiltonian(first, [qml.Y(3), qml.Y(0), qml.Projector([1], wires=[2])]),
                qml.Hamiltonian(second, [qml.Y(3), qml.Y(0) @ qml.X(3), qml.Hadamard(1)]),
            ),
        )
        observable = qml.prod(part, qml.adjoint(part))
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        reconstructed = sum(
            self.scalar_value(coefficient) * self.dense_operator(operator, wires)
            for coefficient, operator in zip(coefficients, operators)
        )
        np.testing.assert_allclose(
            reconstructed,
            self.dense_operator(observable, wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_product_that_autograd_cannot_simplify_is_multiplied_out(self) -> None:
        """``simplify`` adds a complex value into a copy of a real pennylane.numpy weight in place, which
        numpy refuses with a ``TypeError``. ``u = SWAP (-2 Y1 Z2 + 1.5 Y0 X1 + 0 P)`` gives
        ``u u^dagger = 6.25 I``."""
        weights = pnp.array([-2.0, 1.5, 0.0])
        hamiltonian = qml.Hamiltonian(
            weights, [qml.Y(1) @ qml.Z(2), qml.Y(0) @ qml.X(1), qml.Projector([1, 1], wires=[1, 2])]
        )
        part = qml.SWAP(wires=[1, 2]) @ hamiltonian
        observable = part @ qml.adjoint(part)
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert [list(operator.pauli_rep) for operator in operators] == [[PauliWord({})]]
        np.testing.assert_allclose(coefficients, [6.25], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_term_whose_leaves_fail_to_simplify_is_refused(self) -> None:
        """A term that fails to simplify is multiplied out and its leaves simplified on their own, so an
        error of theirs is raised."""
        part = qml.adjoint(qml.Projector([1], wires=[1]))
        with mock.patch.object(type(part), "simplify", side_effect=RuntimeError("in-place")):
            with pytest.raises(RuntimeError, match="in-place"):
                live_pauli_terms(qml.Z(0) + part, keep_non_pauli=True)

    def test_reading_the_terms_queues_no_operator(self) -> None:
        """As ``Sum.terms()``, so that a call inside a circuit adds nothing to its tape."""
        hamiltonian = qml.Hamiltonian([2.0, 0.0], [qml.Z(1), qml.Projector([0], wires=[3])])
        observable = qml.Z(0) + qml.prod(hamiltonian, hamiltonian)
        with qml.queuing.AnnotatedQueue() as queue:
            live_pauli_terms(observable, keep_non_pauli=True)
        assert len(queue.queue) == 0

    def test_a_torch_weight_inside_a_part_kept_whole_stays_differentiable(self) -> None:
        """The part is simplified from a copy of its parameters, which must keep their gradient."""
        weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        observable = qml.Z(0) + qml.prod(qml.Hadamard(1), qml.Hadamard(1), weight * qml.Z(2))
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert isinstance(operators[1], qml.Z)
        coefficients[1].backward()
        np.testing.assert_allclose(float(weight.grad), 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_coefficient_at_the_cutoff_is_dropped_as_sum_terms_drops_it(self) -> None:
        """PennyLane keeps only the summands above its cutoff, so the cutoff itself is dropped."""
        observable = qml.sum(qml.Projector([0], wires=[0]), qml.Z(1), 1.0e-12 * qml.X(2))
        _, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert not any(isinstance(operator, qml.X) for operator in operators)

    @pytest.mark.parametrize("values, kept", [([0.5, 0.0], True), ([1.0e-13, 0.0], False)])
    def test_a_batched_coefficient_is_dropped_only_when_every_entry_is_negligible(
        self, values: list, kept: bool
    ) -> None:
        observable = qml.sum(qml.Projector([0], wires=[0]), qml.s_prod(np.array(values), qml.X(2)))
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert any(isinstance(operator, qml.X) for operator in operators) == kept
        if kept:
            np.testing.assert_allclose(
                coefficients[1], values, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    @pytest.mark.parametrize(
        "make_weight",
        [lambda: 0.5, lambda: torch.tensor(0.5, dtype=torch.float64, requires_grad=True)],
    )
    def test_a_cancelled_term_is_dropped_where_sum_terms_drops_it(self, make_weight) -> None:
        """Without a Pauli representation, ``Sum.terms()`` simplifies and drops what cancels, trainable or not."""
        weight = make_weight()
        observable = qml.sum(qml.Projector([0], wires=[0]), qml.Z(1), weight * qml.X(2), -weight * qml.X(2))
        _, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert not any(isinstance(operator, qml.X) for operator in operators)

    def test_a_cancelled_word_is_kept_when_the_observable_has_a_pauli_representation(self) -> None:
        """``Sum.terms()`` then reads ``pauli_rep``, which keeps it, and the words depend on structure only."""
        _, operators = live_pauli_terms(qml.sum(qml.Z(1), 0.5 * qml.X(2), -0.5 * qml.X(2)), keep_non_pauli=True)
        assert any(isinstance(operator, qml.X) for operator in operators)

    def test_parts_that_simplify_to_the_same_operator_are_merged(self) -> None:
        observable = qml.sum(
            qml.Z(0), qml.Projector([0], wires=[1]) @ qml.Identity(2), -1.0 * qml.Projector([0], wires=[1])
        )
        _, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert not any(isinstance(operator, qml.Projector) for operator in operators)

    def test_an_abstract_coefficient_is_never_dropped(self) -> None:
        """A traced value cannot be compared with the cutoff, so the term is kept."""
        observable = qml.sum(qml.Projector([0], wires=[0]), 0.0 * qml.X(2))
        with mock.patch.object(qml.math, "is_abstract", return_value=True):
            _, operators = live_pauli_terms(observable, keep_non_pauli=True)
        assert any(isinstance(operator, qml.X) for operator in operators)

    def test_an_observable_that_cancels_entirely_has_no_terms(self) -> None:
        projector = qml.Projector([0], wires=[0])
        assert live_pauli_terms(projector - projector, keep_non_pauli=True) == ([], [])

    def test_simplifying_a_part_leaves_the_caller_coefficients_untouched(self) -> None:
        """PennyLane's ``simplify`` merges the two projector terms in place, into a scalar that ``qml.dot``
        unbound as a view of the caller's vector, so a copy is simplified instead."""
        coefficients = torch.tensor([-0.75, -0.3], dtype=torch.float64)
        projector = qml.Projector([1], wires=[1])
        observable = qml.prod(qml.dot(coefficients, [qml.Z(0) + projector, projector]), qml.Identity(2))
        values, operators = live_pauli_terms(observable, keep_non_pauli=True)
        np.testing.assert_array_equal(coefficients.numpy(), [-0.75, -0.3])
        merged = [value for value, operator in zip(values, operators) if isinstance(operator, qml.Projector)]
        np.testing.assert_allclose(merged, [-1.05], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_simplifying_a_part_leaves_the_caller_observable_untouched(self) -> None:
        """Once ``simplify`` has run on a product, PennyLane's ``terms()`` gives a different operator for it."""

        def build() -> qml.operation.Operator:
            weight = torch.tensor(0.0, dtype=torch.float64)
            return qml.Projector([1], wires=[0]) @ qml.s_prod(weight, qml.X(1) @ qml.X(2))

        observable = build()
        live_pauli_terms(observable, keep_non_pauli=True)
        assert [operator.hash for operator in observable.terms()[1]] == [
            operator.hash for operator in build().terms()[1]
        ]

    def test_a_long_sum_chain_is_expanded_in_linear_time(self) -> None:
        """Each term is accumulated once, not once per level of the chain it sits in."""
        words = [qml.prod(qml.X(index % 3), qml.Z((index + 1) % 3)) for index in range(3)]
        observable = 0.5 * words[0]
        n_terms = 600
        for index in range(1, n_terms):
            observable = qml.sum(observable, (0.5 + index) * words[index % 3])
        with mock.patch.object(
            pauli_decomposition_module, "_accumulate", wraps=pauli_decomposition_module._accumulate
        ) as accumulate:
            live_pauli_sentence(observable)
        assert accumulate.call_count <= 4 * n_terms

    def test_a_torch_weight_through_two_i_phases_comes_back_real(self) -> None:
        """``X Y X Y = -I`` passes through an ``i`` phase twice; a complex tensor left with a zero
        imaginary part would make a real expectation complex and its ``backward`` fail."""
        weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        coefficients, _ = live_pauli_terms(weight * (qml.X(0) @ qml.Y(0) @ qml.X(0) @ qml.Y(0)) + qml.Z(1))
        assert coefficients.dtype == torch.float64
        coefficients[0].backward()
        np.testing.assert_allclose(float(weight.grad), -1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_part_kept_whole_twice_is_one_term(self) -> None:
        """The same operator object reached twice merges, rather than being measured twice."""
        projector = qml.Projector([1], wires=[0])
        coefficients, operators = live_pauli_terms(qml.sum(projector, 0.5 * projector), keep_non_pauli=True)
        assert operators == [projector]
        np.testing.assert_allclose(coefficients, [1.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_an_observable_with_no_terms_has_no_coefficients(self) -> None:
        """PennyLane folds a scaled empty Hamiltonian into an empty one, so a sum is used to reach the walk."""
        coefficients, operators = live_pauli_terms(qml.sum(qml.Hamiltonian([], []), qml.Hamiltonian([], [])))
        assert len(coefficients) == len(operators) == 0

    def test_integer_torch_coefficients_are_promoted_to_double(self) -> None:
        coefficients, _ = live_pauli_terms(torch.tensor(2) * qml.Z(0) + torch.tensor(3) * qml.X(1))
        assert coefficients.dtype == torch.float64
        np.testing.assert_allclose(
            coefficients.numpy(), [2.0, 3.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_whole_terms_compare_by_the_identity_of_their_operator(self) -> None:
        """Two keys for one operator object merge; two equal but distinct operators stay apart."""
        projector = qml.Projector([1], wires=[0])
        assert pauli_decomposition_module._WholeTerm(projector) == pauli_decomposition_module._WholeTerm(projector)
        assert pauli_decomposition_module._WholeTerm(projector) != pauli_decomposition_module._WholeTerm(
            qml.Projector([1], wires=[0])
        )

    def test_a_trainable_torch_scale_over_a_numpy_coefficient_is_multiplied(self) -> None:
        """torch cannot multiply a trainable tensor by a numpy array, so numpy values are read as
        python numbers before any arithmetic."""
        weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        observable = qml.s_prod(weight, qml.s_prod(np.array(0.5), qml.Z(0)) + qml.X(1) @ qml.X(2))
        coefficients, _ = live_pauli_terms(observable)
        coefficients.sum().backward()
        np.testing.assert_allclose(
            coefficients.detach().numpy(), [0.35, 0.7], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(float(weight.grad), 1.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_numpy_scale_over_a_torch_coefficient_is_multiplied(self) -> None:
        observable = qml.Hamiltonian(
            [pnp.array(0.7, requires_grad=False)],
            [qml.s_prod(torch.tensor(0.5, dtype=torch.float64), qml.Z(0)) + qml.X(1) @ qml.X(2)],
        )
        coefficients, _ = live_pauli_terms(observable)
        np.testing.assert_allclose(
            coefficients.numpy(), [0.35, 0.7], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_single_precision_numpy_phase_keeps_its_imaginary_part_next_to_torch(self) -> None:
        """``np.complex64`` is read by torch as a real number, so ``0.5 i - 0.5 i`` would come out
        as a spurious ``-0.5 i``; numpy values are therefore read as python numbers first."""
        observable = qml.sum(
            qml.s_prod(np.float32(0.5), qml.prod(qml.X(0), qml.Y(0))),
            qml.s_prod(torch.tensor(0.7, dtype=torch.float64), qml.Z(0)),
            qml.s_prod(np.float32(0.5), qml.prod(qml.Y(0), qml.X(0))),
        )
        coefficients, _ = live_pauli_terms(observable)
        assert coefficients.dtype == torch.float64
        np.testing.assert_allclose(
            coefficients.numpy(), [0.7], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_integer_coefficients_do_not_overflow(self) -> None:
        """Integers are read as floats: ``numpy`` integer arithmetic would wrap around at ``2**63``."""
        observable = qml.prod(*[qml.adjoint(qml.s_prod(10, qml.Z(0))) for _ in range(19)], qml.Z(1))
        coefficients, _ = live_pauli_terms(observable)
        np.testing.assert_allclose(coefficients, [1e19], atol=0.0, rtol=RTOL_SCALAR_COMPARISON)

    def test_single_precision_torch_coefficients_come_back_in_double(self) -> None:
        observable = qml.sum(
            qml.s_prod(torch.tensor(0.5, dtype=torch.float32), qml.X(0)),
            qml.s_prod(torch.tensor(0.25, dtype=torch.float32), qml.Z(0)),
        )
        coefficients, _ = live_pauli_terms(observable)
        assert coefficients.dtype == torch.float64

    def test_a_round_off_imaginary_residue_is_dropped(self) -> None:
        """The cube of a real Hamiltonian with Y terms cancels its cross terms only to about 1e-17."""
        hamiltonian = (
            -0.251 * (qml.Y(0) @ qml.Z(1) @ qml.Z(2)) - 0.291 * qml.Identity(0) + 1.793 * (qml.Z(1) @ qml.X(2))
        )
        coefficients, _ = live_pauli_terms(qml.pow(hamiltonian, 3))
        assert not np.iscomplexobj(coefficients)

    def test_stacked_coefficients_reads_every_interface_into_one_array(self) -> None:
        weight = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
        stacked = stacked_coefficients([0.5, weight, np.float32(0.125), pnp.array(1.0), 2])
        assert isinstance(stacked, torch.Tensor)
        assert stacked.dtype == torch.float64
        np.testing.assert_allclose(
            stacked.detach().numpy(),
            [0.5, 0.25, 0.125, 1.0, 2.0],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        stacked.sum().backward()
        np.testing.assert_allclose(weight.grad.numpy(), [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_single_precision_torch_weights_are_multiplied_in_double(self) -> None:
        """Casting only the final result would keep the float32 rounding of every product on the way."""
        observable = qml.s_prod(torch.tensor(0.5, dtype=torch.float32), qml.s_prod(0.1, qml.Z(0)))
        coefficients, _ = live_pauli_terms(observable)
        assert abs(float(coefficients[0]) - 0.05) <= DOUBLE_PRECISION_ATOL

    def test_a_genuinely_complex_torch_coefficient_stays_complex(self) -> None:
        weight = torch.tensor(0.5, dtype=torch.float64)
        coefficients, _ = live_pauli_terms(qml.s_prod(1j * weight, qml.Z(0)) + qml.X(1))
        assert coefficients.dtype == torch.complex128
        np.testing.assert_allclose(
            coefficients.numpy(), [0.5j, 1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_batched_torch_coefficient_keeps_its_batch_in_the_walk(self) -> None:
        """Only a one-element array is read as a scalar; a batch is carried through as it is."""
        sentence = live_pauli_sentence(torch.tensor([0.5, 0.25], dtype=torch.float32) * qml.Z(0))
        (value,) = sentence.values()
        assert tuple(value.shape) == (2,)
        assert value.dtype == torch.float64

    def test_a_one_element_autograd_tracer_is_read_as_a_scalar(self) -> None:
        """The shape must not depend on whether autograd is tracing."""
        shapes = []

        def total(weights):
            coefficients, _ = live_pauli_terms(qml.sum(qml.s_prod(weights[0:1], qml.X(0)), qml.s_prod(2.0, qml.Z(1))))
            shapes.append(qml.math.shape(coefficients))
            return qml.math.sum(coefficients)

        weights = pnp.array([0.4, 0.1], requires_grad=True)
        total(weights)
        gradient = qml.grad(total)(weights)
        assert shapes == [(2,), (2,)]
        np.testing.assert_allclose(gradient, [1.0, 0.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
