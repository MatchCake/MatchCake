import itertools
from typing import Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane import numpy as pnp
from pennylane.ops import LinearCombination
from pennylane.ops.op_math import Adjoint, ChangeOpBasis, Pow, Prod, SProd, Sum

from matchcake.devices.branch_tensor.pauli_majorana_terms import PauliMajoranaTerms
from matchcake.devices.branch_tensor.wick_reduction import WickReduction
from matchcake.utils.covariance import lift_from_product_state, transition_cov
from matchcake.utils.majorana import MajoranaGetter

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)

# Qubit count of the dense oracles. Three qubits give all 64 Pauli words and every marginal projector
# up to weight 3 while the matrices stay 8x8, so the sweeps below are exhaustive rather than sampled.
N_QUBITS = 3
WIRES = list(range(N_QUBITS))

# Single-qubit Pauli matrices, written out so the dense oracle shares nothing with PennyLane's algebra.
PAULI_MATRICES = {
    "Identity": np.eye(2, dtype=complex),
    "PauliX": np.array([[0, 1], [1, 0]], dtype=complex),
    "PauliY": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "PauliZ": np.array([[1, 0], [0, -1]], dtype=complex),
}

# Separates a coefficient computed in double precision (error ~1e-16) from one that went through a
# float32 or complex64 intermediate (error ~1e-8 on an O(1) value), which PennyLane's cached
# pauli_rep introduces when it combines a torch float64 scalar with a python one.
DOUBLE_PRECISION_ATOL = 1e-12

# Nesting depth past CPython's default recursion limit of 1000, as a qml.sum(H, term) loop over a
# Hamiltonian of that many terms builds.
DEEP_NESTING = 1500

# Parity-marker index D - 1 in the even (2n + 2) lift: the b0 ancilla sits at 2n and the marker at
# 2n + 1, see matchcake.utils.covariance.lift_from_product_state.
MARKER = 2 * N_QUBITS + 1


class TestPauliMajoranaTerms:
    @staticmethod
    def majorana_matrices(n_qubits: int = N_QUBITS) -> list:
        """Dense Majorana operators ``c_0 ... c_{2n-1}`` from MatchCake's own getter.

        The oracle deliberately goes through :class:`~matchcake.utils.majorana.MajoranaGetter`
        rather than through :class:`~matchcake.utils.JordanWigner`, which the module under test
        uses: agreement then pins that the two conventions match, not only that the module is
        self-consistent.

        :param n_qubits: Number of qubits.
        :return: List of ``2n`` dense ``(2**n, 2**n)`` matrices.
        :rtype: list
        """
        getter = MajoranaGetter(n_qubits)
        return [np.asarray(qml.math.toarray(getter[mode]), dtype=complex) for mode in range(2 * n_qubits)]

    @staticmethod
    def pauli_words(wires: Sequence) -> list:
        """Every Pauli word on ``wires``, as ``(label, operator)`` pairs.

        :param wires: Wire labels in qubit order.
        :return: ``4**n`` pairs, the label spelling the word in qubit order.
        :rtype: list
        """
        words = []
        for characters in itertools.product("IXYZ", repeat=len(wires)):
            factors = [getattr(qml, character)(wire) for character, wire in zip(characters, wires) if character != "I"]
            if not factors:
                operator = qml.Identity(wires[0])
            elif len(factors) == 1:
                operator = factors[0]
            else:
                operator = qml.prod(*factors)
            words.append(("".join(characters), operator))
        return words

    @staticmethod
    def is_parity_odd(label: str) -> bool:
        """Whether a Pauli word maps to an odd Majorana monomial.

        Under Jordan-Wigner, ``X`` and ``Y`` each contribute one Majorana beyond the even ``Z``
        strings, and ``Z`` contributes a pair, so the parity is the number of ``X`` and ``Y`` factors.

        :param label: Word spelled in qubit order, e.g. ``"XIZ"``.
        :return: ``True`` for an odd word.
        :rtype: bool
        """
        return sum(character in "XY" for character in label) % 2 == 1

    @staticmethod
    def product_state(seed: int, n_qubits: int = N_QUBITS) -> np.ndarray:
        """Random single-qubit product state, which is not of definite parity.

        Its displacement ``<c_mu>`` is nonzero, so parity-odd words have nonzero expectations and
        only the lifted path can reach them.

        :param seed: Seed of the generator drawing the angles.
        :param n_qubits: Number of qubits.
        :return: Statevector of length ``2**n``.
        :rtype: np.ndarray
        """
        generator = np.random.default_rng(seed)
        vector = np.array([1.0 + 0.0j])
        for _ in range(n_qubits):
            theta = generator.uniform(0.2, 1.3)
            phi = generator.uniform(0.0, 2.0 * np.pi)
            vector = np.kron(vector, np.array([np.cos(theta), np.exp(1j * phi) * np.sin(theta)]))
        return vector

    @staticmethod
    def gaussian_state(seed: int, n_qubits: int = N_QUBITS) -> np.ndarray:
        """Matchgate circuit applied to ``|0...0>``: a parity-even Gaussian state.

        Only parity-preserving gates are used, so the displacement vanishes and the basis path, with
        no marker, is the right frame.

        :param seed: Seed of the generator drawing the angles.
        :param n_qubits: Number of qubits, at least 3.
        :return: Statevector of length ``2**n``.
        :rtype: np.ndarray
        """
        angles = np.random.default_rng(seed).normal(size=6)
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device)
        def statevector():
            qml.IsingXX(angles[0], wires=[0, 1])
            qml.IsingXX(angles[1], wires=[1, 2])
            qml.IsingYY(angles[2], wires=[1, 2])
            qml.IsingXX(angles[3], wires=[0, 1])
            qml.RZ(angles[4], wires=2)
            qml.IsingYY(angles[5], wires=[0, 1])
            return qml.state()

        return np.asarray(statevector(), dtype=complex)

    @staticmethod
    def marginal_probability(
        vector: np.ndarray, bits: Sequence[int], qubits: Sequence[int], n_qubits: int = N_QUBITS
    ) -> float:
        """Probability of reading ``bits`` on ``qubits``, from the statevector, big-endian.

        :param vector: Statevector of length ``2**n``.
        :param bits: Outcome bit of each measured qubit.
        :param qubits: Measured qubit indices, index-aligned with ``bits``.
        :param n_qubits: Number of qubits.
        :return: The marginal probability.
        :rtype: float
        """
        probabilities = np.abs(vector) ** 2
        return float(
            sum(
                probabilities[index]
                for index in range(2**n_qubits)
                if all(((index >> (n_qubits - 1 - qubit)) & 1) == bit for qubit, bit in zip(qubits, bits))
            )
        )

    @staticmethod
    def dense_projector(bits: Sequence[int], qubits: Sequence[int], n_qubits: int = N_QUBITS) -> np.ndarray:
        """``prod_i |y_i><y_i|`` on the measured qubits, identity elsewhere, big-endian.

        :param bits: Outcome bit of each measured qubit.
        :param qubits: Measured qubit indices, index-aligned with ``bits``.
        :param n_qubits: Number of qubits.
        :return: Diagonal ``(2**n, 2**n)`` projector.
        :rtype: np.ndarray
        """
        diagonal = [
            1.0 if all(((index >> (n_qubits - 1 - qubit)) & 1) == bit for qubit, bit in zip(qubits, bits)) else 0.0
            for index in range(2**n_qubits)
        ]
        return np.diag(np.asarray(diagonal, dtype=complex))

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

    @classmethod
    def dense_operator(cls, operator: qml.operation.Operator, wires: Sequence) -> np.ndarray:
        """Dense matrix of an observable, rebuilt by walking its expression tree.

        ``qml.matrix`` evaluates sums and products through ``pauli_rep``, the representation the
        module itself reads, so an oracle built on it cannot see a wrong observable-to-sentence step,
        nor a cached ``pauli_rep`` gone stale. This walks the tree with the coefficients read live and
        the leaves taken from :data:`PAULI_MATRICES`.

        :param operator: Observable built from Pauli words, ``Identity``, sums, scalar products,
            products, integer powers and adjoints.
        :param wires: Device wire labels in qubit order, big-endian.
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
            exponent = float(operator.z)
            assert exponent.is_integer(), "the oracle only takes integer powers"
            return np.linalg.matrix_power(cls.dense_operator(operator.base, wires), int(exponent))
        if isinstance(operator, Adjoint):
            return cls.dense_operator(operator.base, wires).conj().T
        if operator.name == "Identity":
            return np.eye(size, dtype=complex)
        (wire,) = operator.wires
        matrix = np.ones((1, 1), dtype=complex)
        for device_wire in wires:
            factor = PAULI_MATRICES[operator.name] if device_wire == wire else PAULI_MATRICES["Identity"]
            matrix = np.kron(matrix, factor)
        return matrix

    @classmethod
    def dense_monomial(cls, support: Sequence[int], n_qubits: int = N_QUBITS) -> np.ndarray:
        """``c_{s1} c_{s2} ... c_{sw}`` in the given order, as a dense matrix.

        :param support: Majorana mode indices, multiplied left to right.
        :param n_qubits: Number of qubits.
        :return: Dense ``(2**n, 2**n)`` matrix.
        :rtype: np.ndarray
        """
        majoranas = cls.majorana_matrices(n_qubits)
        matrix = np.eye(2**n_qubits, dtype=complex)
        for mode in support:
            matrix = matrix @ majoranas[mode]
        return matrix

    @classmethod
    def reconstruct(
        cls,
        supports: Sequence[Tuple[int, ...]],
        coefficients: Sequence[complex],
        n_qubits: int = N_QUBITS,
        marker: Optional[int] = None,
    ) -> np.ndarray:
        """Dense ``sum_k coefficient_k c_{S_k}``, with the marker stripped from every support.

        The marker is an ancilla mode outside the physical register, so on the physical register
        the monomial it decorates is the support without it.

        :param supports: Supports returned by the module.
        :param coefficients: Index-aligned coefficients returned by the module.
        :param n_qubits: Number of qubits.
        :param marker: Marker index to strip, or ``None``.
        :return: Dense ``(2**n, 2**n)`` operator.
        :rtype: np.ndarray
        """
        operator = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        for support, coefficient in zip(supports, coefficients):
            physical = [mode for mode in support if mode != marker]
            operator = operator + complex(coefficient) * cls.dense_monomial(physical, n_qubits)
        return operator

    @classmethod
    def covariance_and_displacement(
        cls, vector: np.ndarray, n_qubits: int = N_QUBITS
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Majorana covariance ``Lambda = (i/2) <[c_mu, c_nu]>`` and displacement ``d = <c_mu>``.

        The covariance sign is the one reproducing
        :func:`~matchcake.utils.covariance.basis_state_covariance_block`; the opposite sign
        conjugates every Wick value.

        :param vector: Statevector of length ``2**n``.
        :param n_qubits: Number of qubits.
        :return: Real ``(2n, 2n)`` covariance and real ``(2n,)`` displacement, as float64 tensors.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        majoranas = cls.majorana_matrices(n_qubits)
        dim = 2 * n_qubits
        covariance = np.zeros((dim, dim))
        displacement = np.zeros(dim)
        for mu in range(dim):
            displacement[mu] = (vector.conj() @ majoranas[mu] @ vector).real
            for nu in range(dim):
                commutator = majoranas[mu] @ majoranas[nu] - majoranas[nu] @ majoranas[mu]
                covariance[mu, nu] = (0.5j * (vector.conj() @ commutator @ vector)).real
        return torch.as_tensor(covariance, dtype=torch.float64), torch.as_tensor(displacement, dtype=torch.float64)

    @classmethod
    def lifted_transition_covariance(cls, vector: np.ndarray) -> torch.Tensor:
        """Transition covariance of a product state with itself, in the even ``(2n + 2)`` lift.

        :param vector: Product statevector.
        :return: Complex ``(2n + 2, 2n + 2)`` transition covariance.
        :rtype: torch.Tensor
        """
        covariance, displacement = cls.covariance_and_displacement(vector)
        lifted = lift_from_product_state(covariance, displacement)
        return transition_cov(lifted, lifted, marker=MARKER)

    def test_a_bare_pauli_word_is_one_unit_coefficient_term(self) -> None:
        supports, coefficients = PauliMajoranaTerms.from_observable(qml.PauliZ(0), WIRES)
        assert supports == [(0, 1)]
        # Z_0 = -i c_0 c_1
        np.testing.assert_allclose(coefficients, [-1j], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("identity", [qml.Identity(0), qml.Identity(2), qml.Identity()])
    def test_the_identity_is_the_empty_support(self, identity: qml.operation.Operator) -> None:
        supports, coefficients = PauliMajoranaTerms.from_observable(identity, WIRES)
        assert supports == [()]
        np.testing.assert_allclose(coefficients, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_hamiltonian_yields_one_entry_per_term_with_its_coefficient(self) -> None:
        """Literal expected values, worked by hand from ``Z_k = -i c_{2k} c_{2k+1}``.

        ``Z_0 Z_1 = (-i)^2 c_0 c_1 c_2 c_3`` carries ``-1``, ``Z_2`` carries ``-i``, and the
        identity carries ``1``, each times its Pauli coefficient.
        """
        observable = qml.Hamiltonian([0.4, -1.1, 2.0], [qml.Z(0) @ qml.Z(1), qml.Z(2), qml.Identity(0)])
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        decomposition = dict(zip(supports, coefficients))
        assert len(supports) == len(coefficients) == 3
        assert set(decomposition) == {(0, 1, 2, 3), (4, 5), ()}
        np.testing.assert_allclose(
            [decomposition[(0, 1, 2, 3)], decomposition[(4, 5)], decomposition[()]],
            [-0.4, 1.1j, 2.0],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_word_on_a_later_qubit_carries_its_jordan_wigner_string(self) -> None:
        """``X_1 = -i Y_1 Z_1``, so ``X_1 Y_2 = -i (Z_0 Y_1)(Z_0 Z_1 Y_2) = -i c_3 c_5``."""
        supports, coefficients = PauliMajoranaTerms.from_observable(qml.X(1) @ qml.Y(2), WIRES)
        assert supports == [(3, 5)]
        np.testing.assert_allclose(coefficients, [-1j], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_every_pauli_word_is_reconstructed_by_its_monomials(self, n_qubits: int) -> None:
        """Exhaustive dense oracle: ``P == sum_k coefficient_k c_{S_k}`` for every word.

        The marker is passed so that odd words are kept too, and stripped in the reconstruction. Any
        error in the Jordan-Wigner phase, the sort sign, the wire order, or a convention mismatch
        between :class:`~matchcake.utils.JordanWigner` and
        :class:`~matchcake.utils.majorana.MajoranaGetter` shows up as a wrong matrix.
        """
        wires = list(range(n_qubits))
        marker = 2 * n_qubits + 1
        for label, operator in self.pauli_words(wires):
            supports, coefficients = PauliMajoranaTerms.from_observable(operator, wires, marker=marker)
            np.testing.assert_allclose(
                self.reconstruct(supports, coefficients, n_qubits, marker=marker),
                self.dense_operator(operator, wires),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=label,
            )

    def test_odd_words_are_dropped_and_even_words_kept_without_a_marker(self) -> None:
        """Opposite-parity Gaussian states are orthogonal, so on the basis path an odd word vanishes."""
        for label, operator in self.pauli_words(WIRES):
            supports, coefficients = PauliMajoranaTerms.from_observable(operator, WIRES, marker=None)
            expected_terms = 0 if self.is_parity_odd(label) else 1
            assert len(supports) == len(coefficients) == expected_terms, label

    def test_with_a_marker_every_word_is_kept_and_only_odd_ones_take_it_last(self) -> None:
        """The marker goes last: ``c_m c_S = -c_S c_m`` for odd ``S``, so its position is not free."""
        for label, operator in self.pauli_words(WIRES):
            (support,), _ = PauliMajoranaTerms.from_observable(operator, WIRES, marker=MARKER)
            if self.is_parity_odd(label):
                assert support[-1] == MARKER, label
                assert MARKER not in support[:-1], label
            else:
                assert MARKER not in support, label
            assert len(support) % 2 == 0, label

    def test_the_marker_leaves_every_even_word_unchanged(self) -> None:
        """Odd words are not compared here: their coefficient under the marker is pinned by the dense
        reconstruction and by the lifted end-to-end oracle, which is where a sign error shows."""
        for label, operator in self.pauli_words(WIRES):
            if self.is_parity_odd(label):
                continue
            supports, coefficients = PauliMajoranaTerms.from_observable(operator, WIRES, marker=MARKER)
            reference_supports, reference_coefficients = PauliMajoranaTerms.from_observable(operator, WIRES)
            assert supports == reference_supports, label
            np.testing.assert_allclose(
                coefficients,
                reference_coefficients,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
                err_msg=label,
            )

    def test_every_support_is_sorted_and_duplicate_free(self) -> None:
        for label, operator in self.pauli_words(WIRES):
            supports, _ = PauliMajoranaTerms.from_observable(operator, WIRES, marker=MARKER)
            for support in supports:
                assert list(support) == sorted(support), label
                assert len(set(support)) == len(support), label

    def test_a_mixed_parity_hamiltonian_keeps_only_its_even_part_without_a_marker(self) -> None:
        observable = qml.Hamiltonian([1.0, 2.0], [qml.PauliX(0), qml.Z(0) @ qml.Z(1)])
        even_only = PauliMajoranaTerms.from_observable(observable, WIRES, marker=None)
        with_marker = PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)
        assert even_only[0] == [(0, 1, 2, 3)]
        np.testing.assert_allclose(even_only[1], [-2.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        assert sorted(with_marker[0]) == [(0, 1, 2, 3), (0, MARKER)]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_basis_path_reproduces_every_expectation_on_a_gaussian_state(self, seed: int) -> None:
        """End-to-end with :class:`WickReduction`: the uniform phase must not be folded in here.

        On a parity-even Gaussian state every odd word has zero expectation, which is exactly what
        dropping it without a marker returns.
        """
        vector = self.gaussian_state(seed)
        covariance, _ = self.covariance_and_displacement(vector)
        gamma = transition_cov(covariance, covariance)
        for label, operator in self.pauli_words(WIRES):
            supports, coefficients = PauliMajoranaTerms.from_observable(operator, WIRES)
            value = complex(WickReduction.sum_monomials(supports, coefficients, gamma)) if supports else 0.0j
            np.testing.assert_allclose(
                value,
                vector.conj() @ self.dense_operator(operator, WIRES) @ vector,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
                err_msg=label,
            )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_lifted_path_reproduces_every_expectation_on_a_product_state(self, seed: int) -> None:
        """End-to-end through the even lift, odd words included.

        A product state is not of definite parity, so its odd words carry nonzero expectations that
        only the marker can reach. A marker placed first instead of last flips the sign of every one.
        """
        vector = self.product_state(seed)
        gamma = self.lifted_transition_covariance(vector)
        for label, operator in self.pauli_words(WIRES):
            supports, coefficients = PauliMajoranaTerms.from_observable(operator, WIRES, marker=MARKER)
            np.testing.assert_allclose(
                complex(WickReduction.sum_monomials(supports, coefficients, gamma)),
                vector.conj() @ self.dense_operator(operator, WIRES) @ vector,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
                err_msg=label,
            )

    def test_the_odd_words_of_a_product_state_are_not_trivially_zero(self) -> None:
        """Guards the previous test against passing vacuously on a state with no odd content."""
        vector = self.product_state(0)
        odd_values = [
            abs(vector.conj() @ self.dense_operator(operator, WIRES) @ vector)
            for label, operator in self.pauli_words(WIRES)
            if self.is_parity_odd(label)
        ]
        assert max(odd_values) > 0.1

    @pytest.mark.parametrize(
        "observable",
        [
            qml.Hamiltonian([0.3, -0.7, 1.2], [qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Z(2), qml.Z(0)]),
            qml.Z(0) + qml.Z(0),
            qml.s_prod(1j, qml.Z(0)),
            qml.s_prod(-2.5, qml.X(0) @ qml.Y(2)),
            (qml.Z(0) + qml.X(1)) + 2 * (qml.Y(0) @ qml.Y(1)),
            qml.prod(qml.Z(0) + qml.Z(1), qml.X(2)),
            qml.X(0) @ qml.Y(0),
            qml.X(0) @ qml.X(1) @ qml.X(2),
            qml.Hamiltonian([0.5], [-qml.Z(0) @ qml.Z(1)]),
            qml.Hamiltonian([1.0], [2.0 * qml.X(0)]),
            qml.Hamiltonian([1.0], [qml.X(0) @ qml.Y(0)]),
            qml.Hamiltonian([2.0], [qml.Hamiltonian([1.0, 0.5], [qml.X(0), qml.Z(1)])]),
            qml.pow(2.0 * qml.X(0), 2),
            qml.pow(1j * qml.Z(0), 1),
            qml.adjoint(2.0 * qml.X(0)),
            qml.adjoint(qml.X(0) @ qml.Y(0)),
            qml.Hamiltonian([1.0, 0.5], [qml.X(0), qml.Z(1)]) - qml.Hamiltonian([0.3, 2.0], [qml.Y(0), qml.X(1)]),
            qml.Hamiltonian([1.0, 0.5], [qml.X(0), qml.Z(1)]) ** 2,
            qml.adjoint(qml.Hamiltonian([1.0, 0.5], [qml.X(0), qml.Y(1) @ qml.Z(2)])),
            qml.Z(0) + qml.pow(qml.X(1), 0),
            qml.prod(qml.pow(qml.X(0), -1), qml.Z(1)),
            qml.sum(qml.pow(qml.X(0), -1), qml.Z(1)),
            qml.prod(qml.pow(qml.X(0), 2.0), qml.Z(1)),
            qml.sum(qml.Hamiltonian([], []), 0.5 * qml.Z(0), 0.25 * (qml.X(0) @ qml.X(1))),
            0.5 * qml.Hamiltonian([], []),
            qml.pow(2.0 * qml.X(0), -1),
            qml.pow(qml.X(0) @ qml.Y(1), -3),
            qml.adjoint(qml.s_prod(1j, qml.X(0) @ qml.Y(1) @ qml.Z(2))),
            (lambda shared: qml.sum(shared, shared, qml.prod(shared, shared)))(qml.Z(0) @ qml.X(1)),
            qml.change_op_basis(qml.X(0), qml.Hamiltonian([1.0, 0.5], [qml.Z(0), qml.Z(1)])),
        ],
    )
    def test_a_composite_observable_is_reconstructed_on_the_lifted_path(
        self, observable: qml.operation.Operator
    ) -> None:
        """Sums, scalar products, products of sums, a product on a shared wire, and wrappers.

        The Hamiltonian, ``pow`` and ``adjoint`` cases are the regression for terms whose operator
        carries its own scalar or phase (``-Z_0 Z_1``, ``2 X_0``, ``X_0 Y_0 = i Z_0``), which were
        read as bare Pauli words and silently lost that factor, and for nested sums such as
        ``H_1 - H_2``, which raised. The later cases (zero, negative or float powers inside a sum or
        product, an empty Hamiltonian inside a sum, negative powers, and the adjoint of a complex
        multiple) have no ``pauli_rep`` on the whole observable or depend on the operand order and
        on the sign of ``c_S^dagger``.
        """
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients, marker=MARKER),
            self.dense_operator(observable, WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True:UserWarning")
    def test_a_one_element_trainable_weight_is_read_as_a_scalar(self) -> None:
        """``nn.Parameter(torch.randn(1))`` is a common single weight. Its shape-``(1,)`` scalar gives
        the scalar product a batch size, and with it no ``pauli_rep`` on the whole sum."""
        weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
        observable = weight * qml.Z(0) + qml.X(0) @ qml.X(1)
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients, marker=MARKER),
            self.dense_operator(observable, WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("interface", ["torch", "numpy", "autograd"])
    def test_an_in_place_coefficient_change_is_seen_on_a_reused_observable(self, interface: str) -> None:
        """Coefficients are read live, as the m-Pfaffian strategy does for the same reason.

        A Hamiltonian's ``pauli_rep`` is built once at construction, and so are its ``operands``: with
        numpy weights each operand holds a copy of its coefficient, so only ``coeffs`` follows an
        in-place update. Torch slices are views and would not tell the two apart.
        """
        if interface == "torch":
            weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        else:
            weights = np.array([0.5, 0.3]) if interface == "numpy" else pnp.array([0.5, 0.3])
        observable = qml.Hamiltonian(weights, [qml.X(0) @ qml.X(1), qml.Z(0) @ qml.Z(1)])
        PauliMajoranaTerms.from_observable(observable, WIRES)
        with torch.no_grad():
            weights[0] = 5.0
            weights[1] = -1.7
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        expected = 5.0 * self.dense_operator(qml.X(0) @ qml.X(1), WIRES) - 1.7 * self.dense_operator(
            qml.Z(0) @ qml.Z(1), WIRES
        )
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("interface", ["torch", "autograd"])
    def test_building_a_sum_from_an_observable_leaves_its_decomposition_unchanged(self, interface: str) -> None:
        """``H + H`` adds the shared coefficient tensors of the cached ``pauli_rep`` in place, so the
        cached representation of ``H`` itself comes out doubled while ``H.coeffs`` is untouched."""
        weight = torch.tensor(0.5, dtype=torch.float64) if interface == "torch" else pnp.array(0.5)
        observable = qml.Hamiltonian([weight], [qml.Z(0)])
        _ = observable + observable
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert supports == [(0, 1)]
        np.testing.assert_allclose(coefficients, [-0.5j], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda inner: qml.adjoint(inner),
            lambda inner: qml.pow(inner, 1),
            lambda inner: qml.pow(inner, 2),
            lambda inner: qml.Hamiltonian([0.5], [inner]),
            lambda inner: qml.Hamiltonian([1.0], [qml.Z(1)]) - inner,
            lambda inner: qml.sum(qml.Z(1), inner),
            lambda inner: qml.prod(inner, qml.X(1) @ qml.X(2)),
        ],
    )
    @pytest.mark.parametrize("interface", ["torch", "autograd"])
    def test_a_wrapped_observable_is_read_live_after_it_was_reused_in_a_sum(self, wrap, interface: str) -> None:
        """Every level of the tree, not only the top: the cached ``pauli_rep`` of ``A`` is doubled by
        ``A + A``, and ``Sum.terms()``, ``Prod.terms()`` and nested term operators all read that cache.

        The wrapper is built before the reuse, as an observable defined once and evaluated later is.
        """
        weights = torch.tensor([1.0, 0.25], dtype=torch.float64) if interface == "torch" else pnp.array([1.0, 0.25])
        inner = qml.Hamiltonian(weights, [qml.Z(0), qml.X(0) @ qml.Y(2)])
        observable = wrap(inner)
        _ = inner + inner
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients, marker=MARKER),
            self.dense_operator(observable, WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda inner: inner,
            lambda inner: qml.Hamiltonian([1.0], [qml.Z(1)]) - inner,
            lambda inner: qml.sum(qml.Z(1), inner),
            lambda inner: qml.adjoint(inner),
            lambda inner: qml.Hamiltonian([0.5], [inner]),
        ],
    )
    def test_a_wrapped_observable_is_read_live_after_an_in_place_update(self, wrap) -> None:
        """An optimizer updates a coefficient tensor in place; the observable built from it must follow."""
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        observable = wrap(qml.Hamiltonian(weights, [qml.Z(0), qml.X(0) @ qml.X(1)]))
        PauliMajoranaTerms.from_observable(observable, WIRES)
        with torch.no_grad():
            weights[0] = 2.0
            weights[1] = -1.7
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients),
            self.dense_operator(observable, WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_sum_of_dot_products_is_read_live_after_an_in_place_update(self) -> None:
        """``qml.dot`` with tensor coefficients builds a ``Sum`` of scalar products, not a Hamiltonian."""
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        observable = qml.dot(weights, [qml.Z(0), qml.X(0) @ qml.X(1)])
        PauliMajoranaTerms.from_observable(observable, WIRES)
        with torch.no_grad():
            weights[0] = 5.0
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients),
            5.0 * self.dense_operator(qml.Z(0), WIRES) + 0.3 * self.dense_operator(qml.X(0) @ qml.X(1), WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_decomposing_neither_mutates_the_observable_nor_drifts_across_calls(self) -> None:
        """``Sum.terms()`` falls back to simplification for a shape-``(1,)`` scalar and accumulates
        into the caller's tensor in place, so every call through it would add the other terms again."""
        weight = torch.tensor([0.5], dtype=torch.float64)
        observable = weight * qml.Z(0) + 2.0 * qml.Z(0) + qml.pow(qml.X(1), 0)
        expected = 2.5 * self.dense_operator(qml.Z(0), WIRES) + np.eye(2**N_QUBITS)
        for _ in range(3):
            supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
            np.testing.assert_allclose(
                self.reconstruct(supports, coefficients),
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
        np.testing.assert_allclose(weight.numpy(), [0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_nested_torch_scalar_keeps_double_precision(self) -> None:
        """``0.1 * (1j * X_0 Y_0) = 0.1j * c_0 c_1``, computed without a float32 intermediate."""
        observable = qml.Hamiltonian(
            [1.0], [torch.tensor(0.1, dtype=torch.float64) * qml.s_prod(1j, qml.X(0) @ qml.Y(0))]
        )
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert supports == [(0, 1)]
        assert abs(coefficients[0] - 0.1j) <= DOUBLE_PRECISION_ATOL

    @pytest.mark.parametrize(
        "observable",
        [
            np.array([0.75]) * (qml.Z(0) @ qml.Z(1)),
            pnp.array([0.75]) * (qml.Z(0) @ qml.Z(1)),
            qml.Hamiltonian([0.75], [qml.Hamiltonian(np.array([1.0]), [qml.Z(0) @ qml.Z(1)])]),
        ],
    )
    def test_a_one_element_numpy_scalar_is_read_as_a_scalar(self, observable: qml.operation.Operator) -> None:
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert supports == [(0, 1, 2, 3)]
        np.testing.assert_allclose(coefficients, [-0.75], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_an_operator_outside_the_pauli_algebra_is_read_through_its_pauli_representation(self) -> None:
        """``SWAP`` is Hermitian and equals ``(I + X X + Y Y + Z Z) / 2``."""
        observable = qml.SWAP(wires=[0, 1])
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)
        words = [qml.Identity(0), qml.X(0) @ qml.X(1), qml.Y(0) @ qml.Y(1), qml.Z(0) @ qml.Z(1)]
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients, marker=MARKER),
            0.5 * sum(self.dense_operator(word, WIRES) for word in words),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("seed", [0, 1])
    def test_a_random_hamiltonian_is_one_reduction_on_the_lifted_path(self, seed: int) -> None:
        """All 64 words in a single observable, reduced in one batched call."""
        generator = np.random.default_rng(seed)
        words = [operator for _, operator in self.pauli_words(WIRES)]
        weights = generator.normal(size=len(words))
        observable = qml.Hamiltonian(list(weights), words)
        vector = self.product_state(seed + 10)
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)
        np.testing.assert_allclose(
            complex(WickReduction.sum_monomials(supports, coefficients, self.lifted_transition_covariance(vector))),
            vector.conj() @ self.dense_operator(observable, WIRES) @ vector,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [["b", "a", "c"], [7, 3, 5], [2, 0, 1]])
    def test_the_device_wire_order_decides_the_qubit_index(self, wires: list) -> None:
        """Wire labels need not be integers, and the device order, not the label, gives the qubit."""
        for label, operator in self.pauli_words(wires):
            supports, coefficients = PauliMajoranaTerms.from_observable(operator, wires, marker=MARKER)
            np.testing.assert_allclose(
                self.reconstruct(supports, coefficients, marker=MARKER),
                self.dense_operator(operator, wires),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"{wires} {label}",
            )

    def test_a_word_on_a_subset_of_the_wires_lands_on_its_own_qubit(self) -> None:
        supports, coefficients = PauliMajoranaTerms.from_observable(qml.Z(2), WIRES)
        assert supports == [(4, 5)]
        np.testing.assert_allclose(coefficients, [-1j], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "coefficient, expected",
        [
            (torch.tensor(0.75, dtype=torch.float64), 0.75),
            (torch.tensor(0.75, dtype=torch.float32), 0.75),
            (np.float64(0.75), 0.75),
            (np.array(0.75), 0.75),
            (3, 3.0),
            (torch.tensor(0.5 + 0.25j, dtype=torch.complex128), 0.5 + 0.25j),
            (np.complex128(0.5 - 0.25j), 0.5 - 0.25j),
        ],
    )
    def test_a_coefficient_of_any_scalar_type_is_unwrapped_to_a_python_complex(
        self, coefficient: object, expected: complex
    ) -> None:
        """Whatever scalar type the observable carries, the coefficient comes back a python ``complex``
        with its imaginary part intact."""
        observable = qml.Hamiltonian([coefficient], [qml.Z(0) @ qml.Z(1)])
        _, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert type(coefficients[0]) is complex
        np.testing.assert_allclose(coefficients[0], -expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_real_torch_weight_on_a_shared_wire_product_becomes_a_complex_coefficient(self) -> None:
        """``0.5 X_0 Y_0 = 0.5 i Z_0 = 0.5 c_0 c_1``: a real weight turns complex through the product."""
        observable = torch.tensor(0.5, dtype=torch.float64) * (qml.X(0) @ qml.Y(0))
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert supports == [(0, 1)]
        np.testing.assert_allclose(coefficients, [0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_scalar_inside_a_hamiltonian_term_reaches_the_expectation(self) -> None:
        """End-to-end consequence of the regression above, on ``|000>``.

        ``<000| Z_0 Z_1 |000> = 1``, so ``H = 0.5 (-Z_0 Z_1)`` has expectation ``-0.5``. Reading the
        term as the bare word ``Z_0 Z_1`` gives ``+0.5``.
        """
        observable = qml.Hamiltonian([0.5], [-qml.Z(0) @ qml.Z(1)])
        vector = np.zeros(2**N_QUBITS, dtype=complex)
        vector[0] = 1.0
        covariance, _ = self.covariance_and_displacement(vector)
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        np.testing.assert_allclose(
            complex(WickReduction.sum_monomials(supports, coefficients, transition_cov(covariance, covariance))),
            -0.5,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_complex_scalar_coefficient_survives(self) -> None:
        """``1j * Z_0 = 1j * (-i c_0 c_1) = c_0 c_1``: the imaginary part must not be dropped."""
        supports, coefficients = PauliMajoranaTerms.from_observable(qml.s_prod(1j, qml.Z(0)), WIRES)
        assert supports == [(0, 1)]
        np.testing.assert_allclose(coefficients, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True:UserWarning")
    def test_a_trainable_coefficient_is_unwrapped_and_so_carries_no_gradient(self) -> None:
        """Documents current behaviour: the decomposition is a constant with respect to the observable.

        The gradient of an expectation reaches the circuit parameters through the transition
        covariance, never through the observable's coefficients, which come back as python
        ``complex``. A trainable Hamiltonian coefficient is therefore silently treated as a constant.
        """
        coefficient = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
        observable = qml.Hamiltonian([coefficient], [qml.Z(0) @ qml.Z(1)])
        _, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert type(coefficients[0]) is complex

    @pytest.mark.parametrize("marker", [MARKER, np.int64(MARKER), torch.tensor(MARKER)])
    def test_the_outputs_are_plain_python_types(self, marker: object) -> None:
        observable = qml.Hamiltonian([0.5, 0.25, 1.0], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.X(2), qml.X(1)])
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES, marker=marker)
        # X_1 = Z_0 c_2 = -i c_0 c_1 c_2 is odd, so the marker lands in its support.
        assert (0, 1, 2, MARKER) in supports
        assert type(supports) is list and type(coefficients) is list
        for support, coefficient in zip(supports, coefficients):
            assert type(support) is tuple
            assert all(type(mode) is int for mode in support)
            assert type(coefficient) is complex

    @pytest.mark.parametrize(
        "observable",
        [qml.Hermitian(np.diag([1.0, -1.0]), wires=0), qml.Projector([0], wires=0)],
    )
    def test_a_non_pauli_observable_is_refused(self, observable: qml.operation.Operator) -> None:
        with pytest.raises(TypeError, match="no Pauli representation"):
            PauliMajoranaTerms.from_observable(observable, WIRES)

    @pytest.mark.parametrize(
        "observable",
        [qml.Z(5), qml.X(0) @ qml.Y(5), qml.Z(0) + 2.0 * qml.X("a"), qml.SWAP(wires=[0, 7])],
    )
    def test_a_word_on_a_wire_outside_the_device_is_refused(self, observable: qml.operation.Operator) -> None:
        """A word is read over the device wires, so a stray factor would otherwise vanish silently."""
        with pytest.raises(ValueError, match="not in the device wires"):
            PauliMajoranaTerms.from_observable(observable, WIRES, marker=MARKER)

    def test_a_change_of_basis_is_walked_in_operand_order(self) -> None:
        """``S^dagger X S = -Y``. PennyLane builds this node's ``pauli_rep`` in reversed operand order,
        which gives ``+Y``, so the node has to be walked as a product rather than read from its cache."""
        supports, coefficients = PauliMajoranaTerms.from_observable(
            qml.change_op_basis(qml.S(0), qml.X(0)), WIRES, marker=MARKER
        )
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients, marker=MARKER),
            -self.dense_operator(qml.Y(0), WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_change_of_basis_is_read_live_after_an_in_place_update(self) -> None:
        weights = np.array([1.0, 0.5])
        observable = qml.change_op_basis(qml.X(0), qml.Hamiltonian(weights, [qml.Z(0), qml.Z(1)]))
        weights[0] = 3.0
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients),
            -3.0 * self.dense_operator(qml.Z(0), WIRES) + 0.5 * self.dense_operator(qml.Z(1), WIRES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("chain", ["sum", "adjoint"])
    def test_a_tree_nested_past_the_recursion_limit_is_expanded(self, chain: str) -> None:
        """A ``qml.sum(H, term)`` loop nests one level per term, so a long Hamiltonian goes past the
        interpreter's recursion limit. An even-length adjoint chain of ``Z_0`` is ``Z_0`` again."""
        observable = qml.Z(0)
        for _ in range(DEEP_NESTING):
            observable = qml.sum(observable, qml.Z(0)) if chain == "sum" else qml.adjoint(observable)
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        assert supports == [(0, 1)]
        repeats = DEEP_NESTING + 1 if chain == "sum" else 1
        np.testing.assert_allclose(
            coefficients, [-1j * repeats], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_zero_power_is_the_identity_without_reading_its_base(self) -> None:
        """``H^0 = I`` even though the Hadamard itself has no Pauli representation."""
        observable = qml.Z(0) + qml.pow(qml.Hadamard(1), 0)
        supports, coefficients = PauliMajoranaTerms.from_observable(observable, WIRES)
        np.testing.assert_allclose(
            self.reconstruct(supports, coefficients),
            self.dense_operator(qml.Z(0), WIRES) + np.eye(2**N_QUBITS),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "observable",
        [
            qml.exp(qml.X(0), 0.5j),
            qml.CNOT(wires=[0, 1]),
            qml.prod(qml.RX(0.0, wires=0), qml.X(0)),
        ],
    )
    def test_a_node_outside_the_pauli_algebra_is_refused_rather_than_read_from_its_cache(
        self, observable: qml.operation.Operator
    ) -> None:
        """Including a rotation by zero, which PennyLane's simplification would fold to the identity:
        the walk does not simplify, and a parametrized gate is not a Pauli observable."""
        with pytest.raises(TypeError, match="no Pauli representation"):
            PauliMajoranaTerms.from_observable(observable, WIRES)

    def test_device_wires_are_matched_by_hash_like_the_jordan_wigner_lookup(self) -> None:
        """A torch scalar compares equal to an int but does not hash like one, so an ``==`` guard
        would let it through and the lookup would then read every factor as the identity."""
        with pytest.raises(ValueError, match="not in the device wires"):
            PauliMajoranaTerms.from_observable(qml.Z(0) @ qml.Z(1), [torch.tensor(0), torch.tensor(1)])

    @pytest.mark.parametrize("exponent", [1 + 1j, 0.5])
    def test_a_complex_or_fractional_power_is_refused(self, exponent: complex) -> None:
        with pytest.raises(TypeError, match="integer powers"):
            PauliMajoranaTerms.from_observable(qml.pow(qml.X(0), exponent), WIRES)

    def test_the_documented_error_survives_an_operator_pennylane_cannot_print(self) -> None:
        """PennyLane's ``repr`` raises on a composite holding an empty Hamiltonian, which must not
        replace the error being reported."""
        observable = qml.pow(qml.sum(qml.X(0), qml.pow(qml.Hamiltonian([], []), 3)), 0.5)
        with pytest.raises(TypeError, match="integer powers"):
            PauliMajoranaTerms.from_observable(observable, WIRES)

    @pytest.mark.parametrize(
        "base",
        [
            qml.X(0) + qml.Z(1),
            qml.X(0) + qml.Y(0) - qml.Y(0),
            (qml.X(0) + qml.Y(0)) @ (qml.X(0) - qml.Y(0)),
        ],
    )
    def test_a_negative_power_of_a_sum_is_refused(self, base: qml.operation.Operator) -> None:
        """The inverse of a sum of Pauli words is not a finite Pauli sum in general.

        Decided on the structure, so a base whose extra words cancel to a zero coefficient is refused
        too: the words of a decomposition must not depend on coefficient values, or a decomposition
        cached for an observable would no longer fit it once a coefficient changes in place.
        """
        with pytest.raises(TypeError, match="invert a single Pauli word"):
            PauliMajoranaTerms.from_observable(qml.Z(1) + qml.pow(base, -1), WIRES)

    @pytest.mark.parametrize("qubits", [[0], [1, 2], [2, 0], [0, 1, 2], [2, 1, 0], [1, 0, 2]])
    def test_the_projector_expansion_reconstructs_the_dense_projector(self, qubits: list) -> None:
        """Every bit pattern, and qubits given out of order, against ``prod_i |y_i><y_i|``."""
        for bits in itertools.product((0, 1), repeat=len(qubits)):
            supports, coefficients = PauliMajoranaTerms.marginal_projector_terms(bits, qubits)
            assert len(supports) == len(coefficients) == 2 ** len(qubits)
            np.testing.assert_allclose(
                self.reconstruct(supports, coefficients) / 2 ** len(qubits),
                self.dense_projector(bits, qubits),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"{qubits} {bits}",
            )

    @pytest.mark.parametrize("seed", [0, 1])
    def test_the_projector_expansion_gives_every_marginal_probability(self, seed: int) -> None:
        """End-to-end through the even lift, on a state with no definite parity."""
        vector = self.product_state(seed)
        gamma = self.lifted_transition_covariance(vector)
        for weight in range(1, N_QUBITS + 1):
            for qubits in itertools.permutations(range(N_QUBITS), weight):
                for bits in itertools.product((0, 1), repeat=weight):
                    supports, coefficients = PauliMajoranaTerms.marginal_projector_terms(bits, qubits)
                    np.testing.assert_allclose(
                        complex(WickReduction.sum_monomials(supports, coefficients, gamma)) / 2**weight,
                        self.marginal_probability(vector, bits, qubits),
                        atol=ATOL_SCALAR_COMPARISON,
                        rtol=RTOL_SCALAR_COMPARISON,
                        err_msg=f"{qubits} {bits}",
                    )

    def test_every_projector_support_is_sorted_duplicate_free_and_even(self) -> None:
        supports, _ = PauliMajoranaTerms.marginal_projector_terms([1, 0, 1], [2, 0, 1])
        for support in supports:
            assert list(support) == sorted(support)
            assert len(set(support)) == len(support)
            assert len(support) % 2 == 0

    def test_a_projector_on_one_qubit_has_the_literal_expansion(self) -> None:
        """``2 |1><1| = I - Z = I + i c_0 c_1``, worked by hand."""
        supports, coefficients = PauliMajoranaTerms.marginal_projector_terms([1], [0])
        assert supports == [(), (0, 1)]
        np.testing.assert_allclose(coefficients, [1.0, 1j], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_an_empty_projector_is_the_identity(self) -> None:
        supports, coefficients = PauliMajoranaTerms.marginal_projector_terms([], [])
        assert supports == [()]
        np.testing.assert_allclose(coefficients, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "bits, qubits",
        [
            (np.array([1, 0]), np.array([2, 0])),
            (torch.tensor([1, 0]), torch.tensor([2, 0])),
            ([True, False], [2, 0]),
            ((1, 0), (2, 0)),
            (np.array([1, 0], dtype=np.uint8), np.array([2, 0], dtype=np.uint8)),
            (torch.tensor([1, 0], dtype=torch.uint8), torch.tensor([2, 0], dtype=torch.uint8)),
        ],
    )
    def test_the_projector_accepts_any_integer_sequence(self, bits: Sequence, qubits: Sequence) -> None:
        """Unsigned bits included: ``1 - 2 * bit`` wraps to 255 in ``uint8`` unless cast to ``int`` first."""
        reference = PauliMajoranaTerms.marginal_projector_terms([1, 0], [2, 0])
        supports, coefficients = PauliMajoranaTerms.marginal_projector_terms(bits, qubits)
        assert supports == reference[0]
        assert all(type(mode) is int for support in supports for mode in support)
        assert all(type(coefficient) is complex for coefficient in coefficients)
        np.testing.assert_allclose(coefficients, reference[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("bits, qubits", [([0, 1], [0]), ([1], [0, 1])])
    def test_misaligned_bits_and_qubits_are_refused(self, bits: list, qubits: list) -> None:
        """Both directions: with fewer bits, ``zip`` would silently drop the trailing qubits."""
        with pytest.raises(ValueError, match="index-aligned"):
            PauliMajoranaTerms.marginal_projector_terms(bits, qubits)

    @pytest.mark.parametrize("qubits", [[1, 1], [1, 0, 1], [0, 1, 1], torch.tensor([1, 1])])
    def test_a_repeated_measured_qubit_is_refused(self, qubits: Sequence) -> None:
        """The sorted union assumes disjoint supports, so a repeat has to raise, not merge.

        Repeats are checked anywhere in the list and after the cast to ``int``: torch tensors hash
        by identity, so a set of the raw elements would not see the repeat.
        """
        with pytest.raises(ValueError, match="repeated measured qubit"):
            PauliMajoranaTerms.marginal_projector_terms([0] * len(qubits), qubits)
