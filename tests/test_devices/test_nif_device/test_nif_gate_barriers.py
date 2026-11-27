import itertools
import warnings

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.exceptions import DeviceError

import matchcake as mc
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.state_preparation import GaussianStatePrep, ProductState

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
LAYOUTS = [
    (5, [0, 1, 2]),
    (5, [2, 3, 4]),
    (5, [1, 2, 3]),
    (5, [0, 2, 3]),
    (5, [0, 1, 3, 4]),
    (5, [1, 3, 4]),
    (5, [0, 2, 4]),
    (6, [1, 2, 4, 5]),
    (6, [0, 3]),
    (6, [2, 5]),
]


class TestNonInteractingFermionicDeviceGateBarriers:
    r"""
    Gate barriers of the reduced states of a matchgate circuit.

    The references are computed from a dense state vector with plain tensor algebra. The state is a
    Kronecker product of single-qubit states, the gates are applied as dense matrices on the qubits they
    act on, which do not have to be neighbours, and a wire is traced out by never touching it again. An
    expectation value of the reduced state is then the one of the same operator on the kept qubits of
    the dense state.
    """

    @staticmethod
    def random_amplitudes(n_wires: int, is_displaced: bool, seed: int) -> np.ndarray:
        """Draw the single-qubit states of a product state.

        :param n_wires: Number of qubits.
        :type n_wires: int
        :param is_displaced: Whether the qubits are superposed. Otherwise they are basis states.
        :type is_displaced: bool
        :param seed: Seed of the generator.
        :type seed: int
        :return: Amplitudes of shape ``(n_wires, 2)``.
        :rtype: np.ndarray
        """
        rng = np.random.default_rng(seed)
        if not is_displaced:
            return np.eye(2, dtype=complex)[rng.integers(0, 2, size=n_wires)]
        amplitudes = rng.normal(size=(n_wires, 2)) + 1j * rng.normal(size=(n_wires, 2))
        return amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)

    @staticmethod
    def random_gates(pairs, depth: int, seed: int):
        """Build random matchgates on pairs of wires.

        :param pairs: Pairs of wires on which gates are applied, in order, at every layer.
        :param depth: Number of layers.
        :type depth: int
        :param seed: Seed of the random angles.
        :type seed: int
        :return: The list of operations.
        """
        rng = np.random.default_rng(seed)
        operations = []
        for _ in range(depth):
            for pair in pairs:
                operations.append(CompRxRx(rng.normal(size=2), wires=list(pair)))
                operations.append(CompRyRy(rng.normal(size=2), wires=list(pair)))
                operations.append(CompRzRz(rng.normal(size=2), wires=list(pair)))
                if rng.random() < 0.3:
                    operations.append(fSWAP(wires=list(pair)))
        return operations

    @staticmethod
    def neighbours(n_wires: int, barriers=()):
        """Pairs of neighbouring wires that are on the same side of every barrier.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param barriers: Positions of the barriers.
        :return: The list of pairs ``[i, i + 1]``.
        """
        return [[wire, wire + 1] for wire in range(n_wires - 1) if wire + 1 not in barriers]

    @staticmethod
    def product_state_vector(amplitudes: np.ndarray) -> np.ndarray:
        """Dense state vector of a product state.

        :param amplitudes: Amplitudes of shape ``(n, 2)``.
        :type amplitudes: np.ndarray
        :return: State vector of shape ``(2^n,)``.
        :rtype: np.ndarray
        """
        state = np.ones(1, dtype=complex)
        for qubit_amplitudes in amplitudes:
            state = np.kron(state, qubit_amplitudes)
        return state

    @staticmethod
    def apply_gates(state: np.ndarray, operations, qubits) -> np.ndarray:
        """Apply two-qubit gates to a dense state vector.

        :param state: State vector of shape ``(2^n,)``.
        :type state: np.ndarray
        :param operations: Two-wire operations, with wires that are positions in ``qubits``.
        :param qubits: The qubit of the dense state behind every wire position of the operations.
        :return: The evolved state vector.
        :rtype: np.ndarray
        """
        n_qubits = int(np.log2(state.shape[0]))
        tensor = state.reshape([2] * n_qubits)
        for operation in operations:
            first_qubit, second_qubit = (qubits[wire] for wire in operation.wires)
            gate = np.asarray(qml.math.toarray(operation.matrix())).reshape(2, 2, 2, 2)
            tensor = np.tensordot(gate, tensor, axes=([2, 3], [first_qubit, second_qubit]))
            tensor = np.moveaxis(tensor, [0, 1], [first_qubit, second_qubit])
        return tensor.reshape(-1)

    @staticmethod
    def apply_controlled_z(state: np.ndarray, first_qubit: int, second_qubit: int) -> np.ndarray:
        """Apply a controlled-Z gate to a dense state vector.

        :param state: State vector of shape ``(2^n,)``.
        :type state: np.ndarray
        :param first_qubit: First qubit of the gate.
        :type first_qubit: int
        :param second_qubit: Second qubit of the gate.
        :type second_qubit: int
        :return: The state vector with a sign on the amplitudes where both qubits are 1.
        :rtype: np.ndarray
        """
        n_qubits = int(np.log2(state.shape[0]))
        tensor = state.reshape([2] * n_qubits).copy()
        index = [slice(None)] * n_qubits
        index[first_qubit] = index[second_qubit] = 1
        tensor[tuple(index)] *= -1
        return tensor.reshape(-1)

    @staticmethod
    def fswaps_to_the_end(n_wires: int, kept_wires):
        """Build the fSWAP gates that move the wires that are not kept behind the kept ones.

        The kept wires keep their order and end on the first positions.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param kept_wires: The kept wires, in increasing order.
        :return: The list of fSWAP gates on neighbouring positions.
        """
        order = list(range(n_wires))
        operations = []
        for removed_wire in sorted(set(order) - set(kept_wires), reverse=True):
            position = order.index(removed_wire)
            while position + 1 < n_wires and order[position + 1] in kept_wires:
                operations.append(fSWAP(wires=[position, position + 1]))
                order[position], order[position + 1] = order[position + 1], order[position]
                position += 1
        return operations

    @staticmethod
    def word_expval(state: np.ndarray, letters: str, qubits) -> float:
        """Expectation value of a Pauli word in a dense state vector.

        :param state: State vector of shape ``(2^n,)``.
        :type state: np.ndarray
        :param letters: One letter per wire of the reduced register.
        :type letters: str
        :param qubits: The qubit of the dense state behind every wire of the reduced register.
        :return: The expectation value.
        :rtype: float
        """
        n_qubits = int(np.log2(state.shape[0]))
        tensor = state.reshape([2] * n_qubits)
        for letter, qubit in zip(letters, qubits):
            tensor = np.moveaxis(np.tensordot(PAULIS[letter], tensor, axes=([1], [qubit])), 0, qubit)
        return float(np.vdot(state, tensor.reshape(-1)).real)

    @staticmethod
    def probabilities(state: np.ndarray, qubits) -> np.ndarray:
        """Probabilities of the computational-basis outcomes of some qubits of a dense state vector.

        :param state: State vector of shape ``(2^n,)``.
        :type state: np.ndarray
        :param qubits: The measured qubits, in increasing order.
        :return: Probabilities of shape ``(2^k,)``.
        :rtype: np.ndarray
        """
        n_qubits = int(np.log2(state.shape[0]))
        tensor = np.abs(state.reshape([2] * n_qubits)) ** 2
        other_qubits = tuple(qubit for qubit in range(n_qubits) if qubit not in qubits)
        return tensor.sum(axis=other_qubits).reshape(-1)

    @staticmethod
    def observable_of(letters: str):
        """PennyLane observable of a Pauli word given with one letter per wire.

        :param letters: The letters of the word.
        :type letters: str
        :return: The observable.
        """
        factors = [getattr(qml, letter)(wire) for wire, letter in enumerate(letters) if letter != "I"]
        return factors[0] if len(factors) == 1 else qml.prod(*factors)

    @staticmethod
    def is_allowed(letters: str, barriers) -> bool:
        """Tell whether a Pauli word follows the rule of every barrier.

        :param letters: The letters of the word, one per wire.
        :type letters: str
        :param barriers: Positions of the barriers.
        :return: ``True`` when the word has an even number of X and Y at or after every barrier.
        :rtype: bool
        """
        return all(sum(letter in "XY" for letter in letters[barrier:]) % 2 == 0 for barrier in barriers)

    @classmethod
    def prepare(cls, n_wires: int, is_displaced: bool, seed: int = TEST_SEED):
        """Run a first circuit on a device and on a dense state vector.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param is_displaced: Whether the initial product state has superposed qubits.
        :type is_displaced: bool
        :param seed: Seed of the state and of the gates.
        :type seed: int
        :return: The device and the dense state vector.
        """
        amplitudes = cls.random_amplitudes(n_wires, is_displaced, seed)
        operations = cls.random_gates(cls.neighbours(n_wires), 2, seed + 1)
        device = mc.NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires)), *operations])
        state = cls.apply_gates(cls.product_state_vector(amplitudes), operations, list(range(n_wires)))
        return device, state

    @pytest.mark.parametrize("n_wires, kept_wires", LAYOUTS)
    @pytest.mark.parametrize("is_displaced", [False, True])
    def test_every_pauli_word_is_exact_or_refused(self, n_wires, kept_wires, is_displaced):
        r"""
        Reduce the state, apply gates on each side of the barriers, and go over every Pauli word of the
        reduced register: the words that follow the rule of the barriers have the expectation value of
        the dense state, and the others are refused.
        """
        device, state = self.prepare(n_wires, is_displaced)
        reduced_state = device.reduced_state(kept_wires)
        barriers = reduced_state.barriers
        n_kept = len(kept_wires)
        follow_up_operations = self.random_gates(self.neighbours(n_kept, barriers), 2, TEST_SEED + 2)
        state = self.apply_gates(state, follow_up_operations, kept_wires)
        reduced_device = mc.NonInteractingFermionicDevice(wires=max(n_kept, 2))
        reduced_device.apply([reduced_state, *follow_up_operations])
        assert reduced_device.barriers == barriers
        n_allowed = 0
        for letters in itertools.product("IXYZ", repeat=n_kept):
            word = "".join(letters)
            if word == "I" * n_kept:
                continue
            if not self.is_allowed(word, barriers):
                with pytest.raises(DeviceError, match="odd number of X and Y"):
                    reduced_device.exact_expval(self.observable_of(word))
                continue
            n_allowed += 1
            np.testing.assert_allclose(
                float(reduced_device.exact_expval(self.observable_of(word))),
                self.word_expval(state, word, kept_wires),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"word {word} with barriers {barriers}",
            )
        assert n_allowed == 4**n_kept // 2 ** len(barriers) - 1
        np.testing.assert_allclose(
            np.asarray(reduced_device.analytic_probability(wires=range(n_kept))),
            self.probabilities(state, kept_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            np.asarray(reduced_device.analytic_probability(wires=[n_kept - 1])),
            self.probabilities(state, kept_wires[-1:]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "n_wires, kept_wires, is_displaced, barriers, num_params",
        [
            (5, [0, 1, 2], False, (), 1),
            (5, [0, 1, 2], True, (), 2),
            (5, [2, 3, 4], False, (), 1),
            (5, [2, 3, 4], True, (0,), 1),
            (5, [0, 2, 3], False, (1,), 1),
            (5, [0, 2, 3], True, (1,), 2),
            (5, [1, 3, 4], False, (1,), 1),
            (5, [1, 3, 4], True, (0, 1), 1),
            (5, [0, 2, 4], True, (1, 2), 2),
            (6, [1, 2, 4, 5], True, (0, 2), 1),
            (6, [0, 5], False, (1,), 1),
        ],
    )
    def test_barriers_of_a_reduced_state(self, n_wires, kept_wires, is_displaced, barriers, num_params):
        device, _ = self.prepare(n_wires, is_displaced)
        reduced_state = device.reduced_state(kept_wires)
        assert reduced_state.barriers == barriers
        assert reduced_state.num_params == num_params
        removed_wires = [wire for wire in range(n_wires) if wire not in kept_wires]
        assert device.partial_trace(removed_wires).barriers == barriers

    @pytest.mark.parametrize("is_displaced", [False, True])
    def test_a_forbidden_word_differs_without_the_barrier(self, is_displaced):
        device, state = self.prepare(5, is_displaced)
        kept_wires = [0, 2, 3]
        with pytest.warns(UserWarning, match="without gate barriers"):
            unprotected_state = device.reduced_state(kept_wires, gate_barrier=False)
        assert unprotected_state.barriers == ()
        unprotected_device = mc.NonInteractingFermionicDevice(wires=3)
        unprotected_device.apply([unprotected_state])
        differences = []
        for word in ("XXI", "XYI", "YXZ", "YIX"):
            unprotected_expval = float(unprotected_device.exact_expval(self.observable_of(word)))
            differences.append(abs(unprotected_expval - self.word_expval(state, word, kept_wires)))
        assert max(differences) > ATOL_APPROX_COMPARISON
        with pytest.raises(DeviceError, match="odd number of X and Y"):
            protected_device = mc.NonInteractingFermionicDevice(wires=3)
            protected_device.apply([device.reduced_state(kept_wires)])
            protected_device.exact_expval(self.observable_of("XXI"))

    def test_a_gate_across_the_gap_differs_without_the_barrier(self):
        device, state = self.prepare(5, False)
        kept_wires = [0, 2, 3]
        gate = CompRxRx(np.array([0.7, -0.4]), wires=[0, 1])
        state = self.apply_gates(state, [gate], kept_wires)
        with pytest.warns(UserWarning, match="without gate barriers"):
            unprotected_state = device.reduced_state(kept_wires, gate_barrier=False)
        unprotected_device = mc.NonInteractingFermionicDevice(wires=3)
        unprotected_device.apply([unprotected_state, gate])
        difference = np.abs(
            np.asarray(unprotected_device.analytic_probability(wires=range(3))) - self.probabilities(state, kept_wires)
        ).max()
        assert difference > ATOL_APPROX_COMPARISON

    @pytest.mark.parametrize(
        "n_wires, kept_wires",
        [(4, [0, 2, 3]), (4, [2, 3]), (5, [0, 2, 3]), (5, [1, 3]), (5, [0, 2, 4]), (5, [1, 2, 4])],
    )
    @pytest.mark.parametrize("is_displaced", [False, True])
    def test_reduced_state_without_barrier_is_the_reduced_state_after_fswaps(self, n_wires, kept_wires, is_displaced):
        r"""
        Without gate barriers, the reduced state is the one obtained by moving the removed wires behind
        the kept ones with fSWAP gates and tracing them out there, where the trace-out is exact.
        """
        device, _ = self.prepare(n_wires, is_displaced)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            unprotected_state = device.reduced_state(kept_wires, gate_barrier=False)
        device.apply(self.fswaps_to_the_end(n_wires, kept_wires))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            moved_state = device.reduced_state(list(range(len(kept_wires))))
        assert moved_state.barriers == ()
        assert unprotected_state.num_params == moved_state.num_params
        for value, expected in zip(unprotected_state.data, moved_state.data):
            np.testing.assert_allclose(
                qml.math.toarray(value),
                qml.math.toarray(expected),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize(
        "n_wires, kept_wires", [(4, [0, 2, 3]), (4, [2, 3]), (5, [0, 2, 3]), (5, [1, 3]), (5, [1, 2, 4])]
    )
    @pytest.mark.parametrize("is_displaced", [False, True])
    def test_reduced_state_without_barrier_is_the_qubit_state_after_controlled_z_gates(
        self, n_wires, kept_wires, is_displaced
    ):
        r"""
        An fSWAP gate is a SWAP gate times a controlled-Z gate. Without gate barriers, the reduced state
        is therefore the reduced state of the qubits after a controlled-Z gate between every removed wire
        and every kept wire behind it, also when gates act across the removed wires afterwards. These
        gates change the reduced state exactly when it needs a gate barrier.
        """
        device, state = self.prepare(n_wires, is_displaced)
        needs_a_barrier = bool(device.reduced_state(kept_wires).barriers)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            unprotected_state = device.reduced_state(kept_wires, gate_barrier=False)
        untouched_state = state
        for removed_wire in sorted(set(range(n_wires)) - set(kept_wires)):
            for kept_wire in kept_wires:
                if kept_wire > removed_wire:
                    state = self.apply_controlled_z(state, removed_wire, kept_wire)
        n_kept = len(kept_wires)
        gates = self.random_gates(self.neighbours(n_kept), 1, TEST_SEED + 3)
        state = self.apply_gates(state, gates, kept_wires)
        untouched_state = self.apply_gates(untouched_state, gates, kept_wires)
        unprotected_device = mc.NonInteractingFermionicDevice(wires=n_kept)
        unprotected_device.apply([unprotected_state, *gates])
        np.testing.assert_allclose(
            np.asarray(unprotected_device.analytic_probability(wires=range(n_kept))),
            self.probabilities(state, kept_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        differences = []
        for letters in itertools.product("IXYZ", repeat=n_kept):
            word = "".join(letters)
            if word == "I" * n_kept:
                continue
            expval = float(unprotected_device.exact_expval(self.observable_of(word)))
            np.testing.assert_allclose(
                expval,
                self.word_expval(state, word, kept_wires),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"word {word}",
            )
            differences.append(abs(expval - self.word_expval(untouched_state, word, kept_wires)))
        assert (max(differences) > ATOL_APPROX_COMPARISON) is needs_a_barrier

    @pytest.mark.parametrize("entry_point", ["apply", "execute_generator", "qnode"])
    def test_gate_across_a_barrier_is_refused(self, entry_point):
        device, _ = self.prepare(5, False)
        reduced_state = device.reduced_state([0, 1, 3, 4])
        assert reduced_state.barriers == (2,)
        allowed_gates = [CompRxRx(np.array([0.3, 0.2]), wires=[0, 1]), CompRyRy(np.array([0.3, 0.2]), wires=[2, 3])]
        crossing_gate = CompRzRz(np.array([0.3, 0.2]), wires=[1, 2])
        reduced_device = mc.NonInteractingFermionicDevice(wires=4)

        def run(gates):
            if entry_point == "apply":
                reduced_device.reset()
                return reduced_device.apply([reduced_state, *gates])
            if entry_point == "execute_generator":
                return reduced_device.execute_generator(iter([reduced_state, *gates]), reset=True)

            @qml.qnode(reduced_device)
            def circuit():
                qml.apply(reduced_state)
                for gate in gates:
                    qml.apply(gate)
                return qml.probs()

            return circuit()

        run(allowed_gates)
        with pytest.raises(DeviceError, match="acts on both sides of the gate barrier before the device position 2"):
            run([*allowed_gates, crossing_gate])

    def test_gate_on_more_than_two_wires_across_a_barrier_is_refused(self):
        device, _ = self.prepare(5, False)
        reduced_device = mc.NonInteractingFermionicDevice(wires=4)
        sptm = mc.operations.SingleParticleTransitionMatrixOperation(np.eye(6), wires=[0, 1, 2])
        with pytest.raises(DeviceError, match="acts on both sides"):
            reduced_device.apply([device.reduced_state([0, 1, 3, 4]), sptm])
        reduced_device.reset()
        reduced_device.apply([device.reduced_state([0, 1, 2, 4]), sptm])

    def test_a_leading_barrier_does_not_block_gates(self):
        device, state = self.prepare(5, True)
        kept_wires = [2, 3, 4]
        reduced_state = device.reduced_state(kept_wires)
        assert reduced_state.barriers == (0,)
        gates = self.random_gates(self.neighbours(3), 2, TEST_SEED + 3)
        reduced_device = mc.NonInteractingFermionicDevice(wires=3)
        reduced_device.apply([reduced_state, *gates])
        np.testing.assert_allclose(
            np.asarray(reduced_device.analytic_probability(wires=range(3))),
            self.probabilities(self.apply_gates(state, gates, kept_wires), kept_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_barriers_are_reset_with_the_state(self):
        device, _ = self.prepare(5, False)
        reduced_device = mc.NonInteractingFermionicDevice(wires=3)
        assert reduced_device.barriers == ()
        reduced_device.apply([device.reduced_state([0, 2, 3])])
        assert reduced_device.barriers == (1,)
        reduced_device.reset()
        assert reduced_device.barriers == ()
        basis_state = qml.BasisState(np.array([1, 0, 1]), wires=range(3))
        reduced_device.apply([basis_state, CompRxRx(np.array([0.1, 0.2]), wires=[0, 1])])
        assert reduced_device.barriers == ()

    @pytest.mark.parametrize("is_displaced", [False, True])
    def test_observables_made_of_several_words(self, is_displaced):
        device, state = self.prepare(5, is_displaced)
        kept_wires = [0, 2, 3]
        reduced_device = mc.NonInteractingFermionicDevice(wires=3)
        reduced_device.apply([device.reduced_state(kept_wires)])
        words, coefficients = ["ZIZ", "IXX", "ZYX", "IZI"], [0.5, -1.2, 0.7, 0.3]
        expected = sum(
            coefficient * self.word_expval(state, word, kept_wires) for word, coefficient in zip(words, coefficients)
        )
        observables = [self.observable_of(word) for word in words]
        for observable in (qml.Hamiltonian(coefficients, observables), qml.dot(coefficients, observables)):
            np.testing.assert_allclose(
                float(reduced_device.exact_expval(observable)),
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
        batch_values = reduced_device.exact_expval(mc.BatchHamiltonian(coefficients, observables))
        np.testing.assert_allclose(
            float(qml.math.sum(batch_values)), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        with pytest.raises(DeviceError, match="odd number of X and Y"):
            reduced_device.exact_expval(qml.Hamiltonian([1.0, 1.0], [qml.Z(0), qml.X(0) @ qml.X(1)]))
        with pytest.raises(DeviceError, match="odd number of X and Y"):
            reduced_device.exact_expval(mc.BatchHamiltonian([1.0, 1.0], [qml.Z(0), qml.X(0) @ qml.X(1)]))

    def test_hermitian_and_projector_observables(self):
        device, state = self.prepare(5, False)
        kept_wires = [0, 2, 3]
        reduced_device = mc.NonInteractingFermionicDevice(wires=3)
        reduced_device.apply([device.reduced_state(kept_wires)])
        allowed_matrix = np.kron(PAULIS["X"], PAULIS["X"]) + 0.5 * np.kron(PAULIS["Z"], PAULIS["I"])
        np.testing.assert_allclose(
            float(reduced_device.exact_expval(qml.Hermitian(allowed_matrix, wires=[1, 2]))),
            self.word_expval(state, "IXX", kept_wires) + 0.5 * self.word_expval(state, "IZI", kept_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        with pytest.raises(DeviceError, match="odd number of X and Y"):
            reduced_device.exact_expval(qml.Hermitian(np.kron(PAULIS["X"], PAULIS["X"]), wires=[0, 1]))
        np.testing.assert_allclose(
            float(reduced_device.exact_expval(qml.Projector(np.array([1, 0, 1]), wires=[0, 1, 2]))),
            self.probabilities(state, kept_wires)[0b101],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        with pytest.raises(DeviceError, match="no Pauli representation"):
            reduced_device.exact_expval(qml.Projector(np.ones(4) / 2, wires=[0, 1]))

    def test_observable_without_barrier_is_not_checked(self):
        device, state = self.prepare(5, True)
        reduced_device = mc.NonInteractingFermionicDevice(wires=3)
        reduced_device.apply([device.reduced_state([0, 1, 2])])
        assert reduced_device.barriers == ()
        np.testing.assert_allclose(
            float(reduced_device.exact_expval(qml.X(0) @ qml.Z(1))),
            self.word_expval(state, "XZI", [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "kept_wires, observable, is_allowed",
        [
            ([0, 1, 3], qml.X(0) @ qml.X(1), True),
            ([0, 1, 3], qml.Y(0), True),
            ([0, 1, 3], qml.Z(0) @ qml.Z(2), True),
            ([0, 1, 3], qml.X(2), False),
            ([0, 1, 3], qml.X(1) @ qml.X(2), False),
            ([1, 2, 3], qml.Z(0) @ qml.Z(1), True),
            ([1, 2, 3], qml.X(0) @ qml.X(1), False),
        ],
    )
    def test_sampling_in_another_basis(self, kept_wires, observable, is_allowed):
        device, state = self.prepare(4, True)
        reduced_state = device.reduced_state(kept_wires)

        @qml.set_shots(20_000)
        @qml.qnode(mc.NonInteractingFermionicDevice(wires=3))
        def circuit():
            qml.apply(reduced_state)
            return qml.expval(observable)

        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        if not is_allowed:
            with pytest.raises(DeviceError, match="sampled in another basis"):
                circuit()
            return
        letters = ["I"] * 3
        for pauli_word in observable.pauli_rep:
            for wire, letter in pauli_word.items():
                letters[wire] = letter
        np.testing.assert_allclose(
            float(circuit()), self.word_expval(state, "".join(letters), kept_wires), atol=5 * ATOL_APPROX_COMPARISON
        )

    def test_samples_in_the_computational_basis(self):
        device, state = self.prepare(5, True)
        kept_wires = [1, 3, 4]
        reduced_state = device.reduced_state(kept_wires)

        @qml.set_shots(20_000)
        @qml.qnode(mc.NonInteractingFermionicDevice(wires=3))
        def circuit():
            qml.apply(reduced_state)
            return qml.probs(wires=[0, 1, 2])

        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        np.testing.assert_allclose(
            np.asarray(circuit()), self.probabilities(state, kept_wires), atol=5 * ATOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("measurement", [qml.vn_entropy, qml.purity, qml.density_matrix])
    def test_entropy_measurements_and_barriers(self, measurement):
        device, _ = self.prepare(6, False)
        reduced_state = device.reduced_state([0, 1, 3, 4, 5])
        assert reduced_state.barriers == (2,)

        def run(wires, state_prep=reduced_state):
            @qml.qnode(mc.NonInteractingFermionicDevice(wires=5))
            def circuit():
                qml.apply(state_prep)
                return measurement(wires=wires)

            return circuit()

        @qml.qnode(mc.NonInteractingFermionicDevice(wires=6))
        def reference(wires):
            qml.apply(device.state_prep_op)
            qml.apply(device.global_sptm)
            return measurement(wires=wires)

        for wires in ([1, 2], [0, 1, 2, 3, 4], [1, 2, 3]):
            with pytest.raises(DeviceError, match="gate barriers before the positions"):
                run(wires)
        for wires, reference_wires in (([0, 1], [0, 1]), ([2, 3, 4], [3, 4, 5]), ([3], [4])):
            np.testing.assert_allclose(
                qml.math.toarray(run(wires)),
                qml.math.toarray(reference(reference_wires)),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
        displaced_device, _ = self.prepare(5, True)
        with pytest.raises(DeviceError, match="gate barriers before the positions"):
            run([0, 1], displaced_device.reduced_state([1, 2, 3, 4], reduced_wires=range(5)[:4]).to_wires(range(5)))

    def test_chain_of_reductions(self):
        r"""
        Reduce, apply gates, reduce again: the barriers are inherited, merged and dropped, and the
        results still match the dense state.
        """
        device, state = self.prepare(6, True)
        first_kept = [0, 1, 3, 4, 5]
        first_state = device.reduced_state(first_kept)
        assert first_state.barriers == (2,)
        first_gates = self.random_gates(self.neighbours(5, first_state.barriers), 2, TEST_SEED + 4)
        state = self.apply_gates(state, first_gates, first_kept)
        first_device = mc.NonInteractingFermionicDevice(wires=5)
        first_device.apply([first_state, *first_gates])

        # (kept wires of the 5-wire device, expected barriers, expected number of parameters)
        cases = [
            ([0, 1, 2, 3], (2,), 2),
            ([0, 1], (), 2),
            ([0, 2, 3], (1,), 2),
            ([0, 3, 4], (1,), 2),
            ([0, 2, 4], (1, 2), 2),
            ([2, 3, 4], (0,), 1),
            ([1, 2, 3], (0, 1), 1),
            ([3, 4], (0,), 1),
        ]
        for second_kept, barriers, num_params in cases:
            second_state = first_device.reduced_state(second_kept)
            assert second_state.barriers == barriers, second_kept
            assert second_state.num_params == num_params, second_kept
            n_kept = len(second_kept)
            qubits = [first_kept[wire] for wire in second_kept]
            second_gates = self.random_gates(self.neighbours(n_kept, barriers), 1, TEST_SEED + 5)
            final_state = self.apply_gates(state, second_gates, qubits)
            second_device = mc.NonInteractingFermionicDevice(wires=n_kept)
            second_device.apply([second_state, *second_gates])
            for letters in itertools.product("IXYZ", repeat=n_kept):
                word = "".join(letters)
                if word == "I" * n_kept or not self.is_allowed(word, barriers):
                    continue
                np.testing.assert_allclose(
                    float(second_device.exact_expval(self.observable_of(word))),
                    self.word_expval(final_state, word, qubits),
                    atol=ATOL_MATRIX_COMPARISON,
                    rtol=RTOL_MATRIX_COMPARISON,
                    err_msg=f"kept {second_kept}, word {word}",
                )

    def test_inherited_barrier_of_a_state_without_displacement(self):
        device, state = self.prepare(6, False)
        first_device = mc.NonInteractingFermionicDevice(wires=5)
        first_device.apply([device.reduced_state([0, 1, 3, 4, 5])])
        assert first_device.barriers == (2,)
        # No kept wire is left on the left of the barrier and the odd words vanish in the true state.
        second_state = first_device.reduced_state([2, 3, 4])
        assert second_state.barriers == ()
        second_device = mc.NonInteractingFermionicDevice(wires=3)
        second_device.apply([second_state])
        for word in ("XII", "XZY", "XXI", "ZIZ", "YXZ"):
            np.testing.assert_allclose(
                float(second_device.exact_expval(self.observable_of(word))),
                self.word_expval(state, word, [3, 4, 5]),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
        # No kept wire is left on the right of the barrier.
        assert first_device.reduced_state([0, 1]).barriers == ()

    def test_raw_covariance_matrix_and_inherited_barriers(self):
        device, _ = self.prepare(6, False)
        first_device = mc.NonInteractingFermionicDevice(wires=5)
        first_device.apply([device.reduced_state([0, 1, 3, 4, 5])])
        with pytest.raises(ValueError, match="gate_barrier=False"):
            first_device.reduced_covariance_matrix([1, 2])
        with pytest.warns(UserWarning, match="without gate barriers"):
            covariance = first_device.reduced_covariance_matrix([1, 2], gate_barrier=False)
        np.testing.assert_allclose(
            covariance,
            qml.math.toarray(first_device.covariance_matrix)[2:6, 2:6],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            first_device.reduced_covariance_matrix([0, 1])
            first_device.reduced_covariance_matrix([2, 3, 4])
        with pytest.warns(UserWarning, match="without gate barriers"):
            assert first_device.reduced_state([0, 1, 2], gate_barrier=False).barriers == ()

    @pytest.mark.parametrize("is_displaced", [False, True])
    def test_fresh_wires_enter_the_place_of_removed_wires(self, is_displaced):
        r"""
        Continue on the same device, as in the tutorial: the removed wires come back in
        :math:`|0\rangle`. The reference adds fresh qubits to the dense state for them.
        """
        n_wires, kept_wires = 5, [0, 1, 3]
        device, state = self.prepare(n_wires, is_displaced)
        reduced_state = device.partial_trace([2, 4], reduced_wires=kept_wires)
        assert reduced_state.barriers == (2,)
        padded_state = reduced_state.to_wires(range(n_wires))
        assert padded_state.barriers == (3,)
        # The device wires 2 and 4 are fresh qubits, appended to the dense state as qubits 5 and 6.
        qubits = [0, 1, 5, 3, 6]
        dense_state = np.kron(state, np.array([1, 0, 0, 0], dtype=complex))
        allowed_gates = self.random_gates([[0, 1], [1, 2], [3, 4]], 2, TEST_SEED + 6)
        dense_state = self.apply_gates(dense_state, allowed_gates, qubits)
        device.execute_generator(iter([reduced_state, *allowed_gates]), reset=True)
        assert device.barriers == (3,)
        n_checked = 0
        for letters in itertools.product("IXYZ", repeat=n_wires):
            word = "".join(letters)
            if word == "I" * n_wires or not self.is_allowed(word, (3,)):
                continue
            n_checked += 1
            np.testing.assert_allclose(
                float(device.exact_expval(self.observable_of(word))),
                self.word_expval(dense_state, word, qubits),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"word {word}",
            )
        assert n_checked == 4**n_wires // 2 - 1
        np.testing.assert_allclose(
            np.asarray(device.analytic_probability(wires=range(n_wires))),
            self.probabilities(dense_state, [0, 1, 3, 5, 6]).reshape([2] * 5).transpose(0, 1, 3, 2, 4).reshape(-1),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        with pytest.raises(DeviceError, match="acts on both sides"):
            device.execute_generator(iter([reduced_state, CompRxRx(np.array([0.2, 0.1]), wires=[2, 3])]), reset=True)

    def test_fresh_wires_in_front_of_a_leading_barrier(self):
        device, state = self.prepare(4, True)
        reduced_state = device.partial_trace([0, 1], reduced_wires=[2, 3])
        assert reduced_state.barriers == (0,)
        gates = self.random_gates([[0, 1], [2, 3]], 2, TEST_SEED + 7)
        device.execute_generator(iter([reduced_state, *gates]), reset=True)
        assert device.barriers == (2,)
        qubits = [4, 5, 2, 3]
        dense_state = self.apply_gates(np.kron(state, np.array([1, 0, 0, 0], dtype=complex)), gates, qubits)
        for word in ("XXII", "ZIZZ", "XYXX", "IIXY", "YXZZ", "IZYY"):
            np.testing.assert_allclose(
                float(device.exact_expval(self.observable_of(word))),
                self.word_expval(dense_state, word, qubits),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"word {word}",
            )
        with pytest.raises(DeviceError, match="odd number of X and Y"):
            device.exact_expval(self.observable_of("IIXI"))
        with pytest.raises(DeviceError, match="acts on both sides"):
            device.execute_generator(iter([reduced_state, CompRxRx(np.array([0.2, 0.1]), wires=[1, 2])]), reset=True)

    def test_leading_barrier_survives_fresh_wires_and_a_second_reduction(self):
        device, state = self.prepare(4, True)
        reduced_state = device.partial_trace([0, 1], reduced_wires=[2, 3])
        assert reduced_state.barriers == (0,)
        assert reduced_state.odd_part_dropped
        gates = self.random_gates([[0, 1], [2, 3]], 2, TEST_SEED + 7)
        device.execute_generator(iter([reduced_state, *gates]), reset=True)
        assert device.barriers == (2,)
        second = device.reduced_state([2, 3])
        assert second.barriers == (0,)
        assert second.odd_part_dropped
        assert second.num_params == 1
        qubits = [4, 5, 2, 3]
        dense_state = self.apply_gates(np.kron(state, np.array([1, 0, 0, 0], dtype=complex)), gates, qubits)
        small_device = mc.NonInteractingFermionicDevice(wires=2)
        small_device.apply([second])
        for word in ("ZZ", "XX", "XY", "ZI"):
            np.testing.assert_allclose(
                float(small_device.exact_expval(self.observable_of(word))),
                self.word_expval(dense_state, "II" + word, qubits),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
                err_msg=f"word {word}",
            )
        assert abs(self.word_expval(dense_state, "IIXI", qubits)) > ATOL_APPROX_COMPARISON
        with pytest.raises(DeviceError, match="odd number of X and Y"):
            small_device.exact_expval(self.observable_of("XI"))
        third = small_device.reduced_state([0, 1])
        assert third.barriers == (0,)
        assert third.odd_part_dropped

    def test_mark_of_a_dropped_displacement_ends_with_the_barriers(self):
        device, _ = self.prepare(4, True)
        reduced_state = device.partial_trace([0, 1], reduced_wires=[2, 3])
        device.execute_generator(iter([reduced_state]), reset=True)
        fresh_wires = device.reduced_state([0, 1])
        assert fresh_wires.barriers == ()
        assert not fresh_wires.odd_part_dropped
        unprotected = device.reduced_state([2, 3], gate_barrier=False)
        assert not unprotected.odd_part_dropped

    def test_inherited_barrier_of_a_displaced_state_becomes_a_leading_barrier(self):
        device, _ = self.prepare(5, True)
        reduced_state = device.reduced_state([0, 2, 3])
        assert reduced_state.barriers == (1,)
        assert not reduced_state.odd_part_dropped
        small_device = mc.NonInteractingFermionicDevice(wires=3)
        small_device.apply([reduced_state])
        second = small_device.reduced_state([1, 2])
        assert second.barriers == (0,)
        assert second.odd_part_dropped
        assert second.num_params == 1

    @pytest.mark.parametrize("measurement", [qml.vn_entropy, qml.purity, qml.density_matrix])
    def test_entropy_measurements_after_a_dropped_displacement(self, measurement):
        device, _ = self.prepare(4, True)
        reduced_state = device.partial_trace([0, 1], reduced_wires=[2, 3])

        def run(wires):
            @qml.qnode(mc.NonInteractingFermionicDevice(wires=4))
            def circuit():
                qml.apply(reduced_state)
                return measurement(wires=wires)

            return circuit()

        for wires in ([2, 3], [2], [1, 2], [3]):
            with pytest.raises(DeviceError, match="gate barriers"):
                run(wires)
        assert np.all(np.isfinite(qml.math.toarray(run([0, 1]))))

    def test_sampling_in_another_basis_with_two_barriers(self):
        device, _ = self.prepare(6, False)
        reduced_state = device.reduced_state([0, 1, 3, 5])
        assert reduced_state.barriers == (2, 3)

        def run(observable):
            @qml.set_shots(10)
            @qml.qnode(mc.NonInteractingFermionicDevice(wires=4))
            def circuit():
                qml.apply(reduced_state)
                return qml.expval(observable)

            return circuit()

        assert np.isfinite(float(run(qml.X(0) @ qml.X(1))))
        for observable in (qml.X(2) @ qml.X(3), qml.X(1) @ qml.X(2), qml.X(3)):
            with pytest.raises(DeviceError):
                run(observable)

    def test_reduced_state_measurement_chain_with_gradients(self):
        r"""
        Chain two QNodes with :func:`matchcake.reduced_state`: the value matches the dense state and the
        gradient with respect to the angles of the product state matches finite differences.
        """
        n_wires, kept_wires = 5, [0, 2, 3]
        first_gates = self.random_gates(self.neighbours(n_wires), 2, TEST_SEED + 8)
        follow_up_gates = self.random_gates([[1, 2]], 1, TEST_SEED + 9)
        observable = qml.Y(0) @ qml.Z(2) + 0.5 * qml.X(1) @ qml.Y(2)

        def amplitudes_of(angles):
            zero_amplitudes = torch.cos(angles[:, 0] / 2) + 0j
            one_amplitudes = torch.exp(1j * angles[:, 1]) * torch.sin(angles[:, 0] / 2)
            return torch.stack([zero_amplitudes, one_amplitudes], dim=-1)

        @qml.qnode(mc.NonInteractingFermionicDevice(wires=n_wires), interface="torch")
        def first_circuit(angles):
            ProductState(amplitudes_of(angles), wires=range(n_wires))
            for gate in first_gates:
                qml.apply(gate)
            return mc.reduced_state(wires=kept_wires)

        @qml.qnode(mc.NonInteractingFermionicDevice(wires=3), interface="torch")
        def follow_up_circuit(state_prep):
            qml.apply(state_prep)
            for gate in follow_up_gates:
                qml.apply(gate)
            return qml.expval(observable)

        def chain(angles):
            return follow_up_circuit(first_circuit(angles))

        angles = torch.tensor(np.random.default_rng(TEST_SEED).uniform(0.3, 2.5, size=(n_wires, 2)), requires_grad=True)
        reduced_state = first_circuit(angles)
        assert isinstance(reduced_state, GaussianStatePrep)
        assert reduced_state.barriers == (1,)
        assert reduced_state.num_params == 2
        value = chain(angles)
        dense_state = self.product_state_vector(amplitudes_of(angles).detach().numpy())
        dense_state = self.apply_gates(dense_state, first_gates, list(range(n_wires)))
        dense_state = self.apply_gates(dense_state, follow_up_gates, kept_wires)
        expected = self.word_expval(dense_state, "YIZ", kept_wires)
        expected += 0.5 * self.word_expval(dense_state, "IXY", kept_wires)
        np.testing.assert_allclose(value.item(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        (gradient,) = torch.autograd.grad(value, angles)
        assert gradient.abs().max() > ATOL_APPROX_COMPARISON
        step = 1e-6
        for index in [(0, 0), (1, 1), (2, 0), (4, 1)]:
            shift = torch.zeros_like(angles)
            shift[index] = step
            with torch.no_grad():
                finite_difference = (chain(angles + shift) - chain(angles - shift)) / (2 * step)
            np.testing.assert_allclose(gradient[index].item(), finite_difference.item(), atol=ATOL_APPROX_COMPARISON)
