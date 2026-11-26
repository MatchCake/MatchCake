import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.state_preparation import GaussianStatePrep, PlusState, ProductState

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

N_WIRES = 5
PAULIS = {
    "I": torch.eye(2, dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}


class TestNonInteractingFermionicDeviceReducedProductState:
    r"""
    Reduced states of a matchgate circuit started from a product state with superposed qubits.

    The references are computed from the dense state vector with plain tensor algebra: the state is
    built as a Kronecker product, the gates are applied as dense matrices and the expectation values are
    traces with Kronecker products of Pauli matrices.
    """

    @staticmethod
    def brick_wall(wires, depth: int, seed: int):
        """Build a random brick-wall matchgate circuit.

        :param wires: Wires of the circuit, in register order.
        :param depth: Number of layers.
        :type depth: int
        :param seed: Seed of the random angles.
        :type seed: int
        :return: The list of operations.
        """
        rng = np.random.default_rng(seed)
        wires = list(wires)
        operations = []
        for _ in range(depth):
            for position in range(len(wires) - 1):
                pair = [wires[position], wires[position + 1]]
                operations.append(CompRxRx(rng.normal(size=2), wires=pair))
                operations.append(CompRyRy(rng.normal(size=2), wires=pair))
                operations.append(CompRzRz(rng.normal(size=2), wires=pair))
            operations.append(fSWAP(wires=[wires[0], wires[1]]))
        return operations

    @staticmethod
    def amplitudes_from_angles(angles: torch.Tensor) -> torch.Tensor:
        """Amplitudes of the qubits of a product state from their Bloch angles.

        :param angles: Polar and azimuthal angles, shape ``(..., n, 2)``.
        :type angles: torch.Tensor
        :return: Amplitudes of shape ``(..., n, 2)``.
        :rtype: torch.Tensor
        """
        theta, phi = angles[..., 0], angles[..., 1]
        return torch.stack([torch.cos(theta / 2) + 0j, torch.exp(1j * phi) * torch.sin(theta / 2)], dim=-1)

    @staticmethod
    def random_angles(n_wires: int = N_WIRES, batch_size=None, seed: int = TEST_SEED) -> torch.Tensor:
        """Draw the Bloch angles of a random product state without any qubit in a basis state.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param batch_size: Optional leading batch size.
        :param seed: Seed of the random generator.
        :type seed: int
        :return: Angles of shape ``(n_wires, 2)`` or ``(batch_size, n_wires, 2)``.
        :rtype: torch.Tensor
        """
        rng = np.random.default_rng(seed)
        shape = (n_wires, 2) if batch_size is None else (batch_size, n_wires, 2)
        return torch.tensor(rng.uniform(0.3, 2.8, size=shape), dtype=torch.float64)

    @staticmethod
    def kron(matrices) -> torch.Tensor:
        """Kronecker product of a list of matrices, the first one acting on the first wire.

        :param matrices: The matrices.
        :return: Their Kronecker product.
        :rtype: torch.Tensor
        """
        result = torch.ones((1, 1), dtype=torch.complex128)
        for matrix in matrices:
            result = torch.kron(result, matrix)
        return result

    @classmethod
    def pauli_word(cls, letters: str) -> torch.Tensor:
        """Dense matrix of a Pauli word given with one letter per wire.

        :param letters: The letters of the word, for example ``"XIZ"``.
        :type letters: str
        :return: Matrix of shape ``(2^n, 2^n)``.
        :rtype: torch.Tensor
        """
        return cls.kron([PAULIS[letter] for letter in letters])

    @classmethod
    def majoranas(cls, n_wires: int):
        """Dense Jordan-Wigner Majorana operators of a register.

        :param n_wires: Number of wires.
        :type n_wires: int
        :return: The list of the ``2 * n_wires`` operators.
        """
        operators = []
        for wire in range(n_wires):
            for letter in ("X", "Y"):
                operators.append(cls.pauli_word("Z" * wire + letter + "I" * (n_wires - wire - 1)))
        return operators

    @classmethod
    def state_vector(cls, amplitudes: torch.Tensor, operations) -> torch.Tensor:
        """Dense state vector of a product state evolved by gates on neighbouring wires.

        :param amplitudes: Amplitudes of the qubits, shape ``(n, 2)``.
        :type amplitudes: torch.Tensor
        :param operations: Operations on wires ``[j, j + 1]`` of the register ``range(n)``.
        :return: State vector of shape ``(2^n,)``.
        :rtype: torch.Tensor
        """
        n_wires = amplitudes.shape[0]
        state = torch.ones(1, dtype=torch.complex128)
        for wire in range(n_wires):
            state = torch.kron(state, amplitudes[wire].to(torch.complex128))
        for operation in operations:
            first_wire, second_wire = operation.wires.tolist()
            assert second_wire == first_wire + 1
            gate = torch.as_tensor(qml.math.toarray(operation.matrix()), dtype=torch.complex128)
            identities_before = [PAULIS["I"]] * first_wire
            identities_after = [PAULIS["I"]] * (n_wires - second_wire - 1)
            state = cls.kron(identities_before + [gate] + identities_after) @ state
        return state

    @staticmethod
    def reduced_density_matrix(state: torch.Tensor, kept_wires) -> torch.Tensor:
        """Reduced density matrix of the qubits of a pure state.

        :param state: State vector of shape ``(2^n,)``.
        :type state: torch.Tensor
        :param kept_wires: Positions of the kept wires, in increasing order.
        :return: Density matrix of shape ``(2^k, 2^k)``.
        :rtype: torch.Tensor
        """
        n_wires = int(np.log2(state.shape[0]))
        kept_wires = list(kept_wires)
        removed_wires = [wire for wire in range(n_wires) if wire not in kept_wires]
        tensor = state.reshape([2] * n_wires).permute(kept_wires + removed_wires)
        matrix = tensor.reshape(2 ** len(kept_wires), 2 ** len(removed_wires))
        return matrix @ matrix.conj().T

    @classmethod
    def covariance_of(cls, expectation, operators) -> np.ndarray:
        """Covariance matrix of a state from the expectation values of pairs of Majorana operators.

        :param expectation: Callable that returns the expectation value of a dense operator.
        :param operators: The Majorana operators.
        :return: Real antisymmetric matrix.
        :rtype: np.ndarray
        """
        size = len(operators)
        covariance = np.zeros((size, size))
        for row in range(size):
            for column in range(size):
                if row != column:
                    covariance[row, column] = (1j * expectation(operators[row] @ operators[column])).real.item()
        return covariance

    @pytest.fixture
    def angles(self) -> torch.Tensor:
        return self.random_angles()

    @pytest.fixture
    def operations(self):
        return self.brick_wall(range(N_WIRES), 2, TEST_SEED)

    @pytest.fixture
    def state(self, angles, operations) -> torch.Tensor:
        return self.state_vector(self.amplitudes_from_angles(angles), operations)

    @pytest.fixture
    def prepared_device(self, angles, operations) -> mc.NonInteractingFermionicDevice:
        """A device holding the product state evolved by the reference circuit."""
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply([ProductState(self.amplitudes_from_angles(angles), wires=range(N_WIRES))] + operations)
        return device

    def test_displacement_vector_of_the_device(self, prepared_device, state):
        expected = [(state.conj() @ operator @ state).real.item() for operator in self.majoranas(N_WIRES)]
        assert np.abs(expected).max() > ATOL_MATRIX_COMPARISON
        np.testing.assert_allclose(
            qml.math.toarray(prepared_device.displacement_vector),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_displacement_vector_rejects_a_non_gaussian_state_preparation(self, prepared_device):
        dense_state = np.zeros(2**N_WIRES)
        dense_state[0] = 1.0
        prepared_device.apply_state_prep(qml.StatePrep(dense_state, wires=range(N_WIRES)), index=0)
        with pytest.raises(ValueError, match="Displacement vector can only be computed"):
            prepared_device.displacement_vector

    @pytest.mark.parametrize("kept_wires", [[0], [0, 1], [0, 1, 2], [3, 4], [2, 3, 4], [1, 2, 3], [4], [0, 1, 2, 3, 4]])
    def test_reduced_covariance_matrix_on_a_contiguous_block(self, prepared_device, state, kept_wires):
        density_matrix = self.reduced_density_matrix(state, kept_wires)
        expected = self.covariance_of(
            lambda operator: torch.trace(density_matrix @ operator), self.majoranas(len(kept_wires))
        )
        np.testing.assert_allclose(
            qml.math.toarray(prepared_device.reduced_covariance_matrix(kept_wires)),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("kept_wires", [[0, 2], [1, 3], [0, 2, 4], [1, 4]])
    def test_fermionic_reduced_covariance_matrix(self, prepared_device, state, kept_wires):
        full_majoranas = self.majoranas(N_WIRES)
        kept_majoranas = [full_majoranas[2 * wire + offset] for wire in kept_wires for offset in (0, 1)]
        expected = self.covariance_of(lambda operator: state.conj() @ operator @ state, kept_majoranas)
        with pytest.raises(ValueError, match="fermionic=True"):
            prepared_device.reduced_covariance_matrix(kept_wires)
        np.testing.assert_allclose(
            qml.math.toarray(prepared_device.reduced_covariance_matrix(kept_wires, fermionic=True)),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("kept_wires", [[0, 1], [2, 3, 4], [1, 2, 3]])
    def test_covariance_matrix_measurement(self, angles, operations, state, kept_wires):
        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES))
        def circuit():
            ProductState(self.amplitudes_from_angles(angles), wires=range(N_WIRES))
            for operation in operations:
                qml.apply(operation)
            return mc.covariance_matrix(wires=kept_wires)

        density_matrix = self.reduced_density_matrix(state, kept_wires)
        expected = self.covariance_of(
            lambda operator: torch.trace(density_matrix @ operator), self.majoranas(len(kept_wires))
        )
        np.testing.assert_allclose(
            qml.math.toarray(circuit()), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("measurement", [qml.vn_entropy, qml.purity, qml.density_matrix])
    @pytest.mark.parametrize("wires", [[0, 1], [3, 4], [0, 1, 2, 3, 4]])
    def test_measurements_of_the_state_are_rejected(self, measurement, wires):
        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES))
        def circuit():
            PlusState(wires=range(N_WIRES))
            return measurement(wires=wires)

        with pytest.raises(NotImplementedError, match="without displacement"):
            circuit()

    @pytest.mark.parametrize("n_kept", [1, 2, 3, 5])
    def test_reduced_state_on_the_first_wires(self, prepared_device, state, n_kept):
        kept_wires = list(range(n_kept))
        density_matrix = self.reduced_density_matrix(state, kept_wires)
        reduced_majoranas = self.majoranas(n_kept)
        reduced_state = prepared_device.reduced_state(kept_wires)
        assert isinstance(reduced_state, GaussianStatePrep)
        assert reduced_state.num_params == 2
        assert reduced_state.wires.tolist() == kept_wires
        np.testing.assert_allclose(
            qml.math.toarray(reduced_state.covariance_matrix),
            self.covariance_of(lambda operator: torch.trace(density_matrix @ operator), reduced_majoranas),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            qml.math.toarray(reduced_state.displacement_vector),
            [torch.trace(density_matrix @ operator).real.item() for operator in reduced_majoranas],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_partial_trace_is_the_complement_of_reduced_state(self, prepared_device):
        traced = prepared_device.partial_trace([3, 4], reduced_wires=["a", "b", "c"])
        kept = prepared_device.reduced_state([0, 1, 2])
        assert traced.wires.tolist() == ["a", "b", "c"]
        for traced_parameter, kept_parameter in zip(traced.data, kept.data, strict=True):
            np.testing.assert_allclose(
                qml.math.toarray(traced_parameter),
                qml.math.toarray(kept_parameter),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize(
        "kept_wires, fermionic",
        [([1, 2], False), ([3, 4], False), ([4], False), ([1, 2, 3, 4], False), ([0, 2], True), ([1, 0], True)],
    )
    def test_reduced_state_away_from_the_first_wires_is_rejected(self, prepared_device, kept_wires, fermionic):
        with pytest.raises(NotImplementedError, match="first wires of the device"):
            prepared_device.reduced_state(kept_wires, fermionic=fermionic)

    def test_partial_trace_of_the_first_wires_is_rejected(self, prepared_device):
        with pytest.raises(NotImplementedError, match="drop_displacement=True"):
            prepared_device.partial_trace([0, 1])

    @pytest.mark.parametrize("kept_wires", [[0, 1], [1, 2], [3, 4], [2, 3, 4], [1, 2, 3, 4]])
    @pytest.mark.parametrize("letters", ["ZI", "ZZ", "XX", "XY", "YX", "YY", "XI"])
    def test_follow_up_circuit_without_the_displacement_vector(
        self, prepared_device, operations, angles, kept_wires, letters
    ):
        r"""
        Reduce the state to a block of wires without its displacement vector, run a second circuit from
        it, and compare with the same second circuit applied to those wires of the full system. The
        probabilities and the words with an even number of X and Y are exact, the other words vanish.
        """
        n_kept = len(kept_wires)
        reduced_state = prepared_device.reduced_state(kept_wires, drop_displacement=True)
        assert reduced_state.num_params == 1
        observable = qml.prod(*[getattr(qml, letter)(wire) for wire, letter in enumerate(letters) if letter != "I"])
        follow_up_operations = self.brick_wall(range(n_kept), 1, TEST_SEED + 2)

        @qml.qnode(qml.device("nif.qubit", wires=n_kept))
        def follow_up():
            qml.apply(reduced_state)
            for operation in follow_up_operations:
                qml.apply(operation)
            return qml.expval(observable), qml.probs()

        final_state = self.state_vector(
            self.amplitudes_from_angles(angles), operations + self.brick_wall(kept_wires, 1, TEST_SEED + 2)
        )
        full_letters = ["I"] * N_WIRES
        for wire, letter in zip(kept_wires, letters):
            full_letters[wire] = letter
        word = self.pauli_word("".join(full_letters))
        exact_expval = (final_state.conj() @ word @ final_state).real.item()
        is_even = (letters.count("X") + letters.count("Y")) % 2 == 0
        if not is_even:
            assert abs(exact_expval) > ATOL_APPROX_COMPARISON
        expected_probabilities = torch.diagonal(self.reduced_density_matrix(final_state, kept_wires)).real.numpy()
        expval, probabilities = follow_up()
        np.testing.assert_allclose(
            expval, exact_expval if is_even else 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            probabilities, expected_probabilities, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_reporter_snippet(self):
        n_wires = 4
        device = mc.NIFDevice(wires=n_wires)
        amplitudes = self.amplitudes_from_angles(self.random_angles(n_wires))
        device.apply(ProductState(state=amplitudes, wires=device.wires))
        reduced_state = device.partial_trace([0, 1], drop_displacement=True)
        assert reduced_state.num_params == 1
        np.testing.assert_allclose(
            qml.math.toarray(reduced_state.covariance_matrix),
            qml.math.toarray(ProductState(amplitudes[2:], wires=[0, 1]).covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_kept", [2, 3, 4])
    @pytest.mark.parametrize(
        "letters",
        ["X", "Y", "YZ", "XX", "XY", "Z", "ZZ"],
    )
    def test_follow_up_circuit_on_the_reduced_state(self, prepared_device, state, operations, angles, n_kept, letters):
        r"""
        Reduce the state to the first wires, run a second circuit on a smaller device from the reduced
        state, and compare with the same second circuit applied to the first wires of the full system.
        The words with an odd number of X and Y need the displacement vector.
        """
        kept_wires = list(range(n_kept))
        reduced_state = prepared_device.reduced_state(kept_wires)
        follow_up_operations = self.brick_wall(kept_wires, 1, TEST_SEED + 2)
        observable = qml.prod(*[getattr(qml, letter)(wire) for wire, letter in enumerate(letters)])

        @qml.qnode(qml.device("nif.qubit", wires=n_kept))
        def follow_up():
            GaussianStatePrep(
                reduced_state.covariance_matrix,
                wires=kept_wires,
                displacement_vector=reduced_state.displacement_vector,
            )
            for operation in follow_up_operations:
                qml.apply(operation)
            return qml.expval(observable), qml.probs()

        final_state = self.state_vector(self.amplitudes_from_angles(angles), operations + follow_up_operations)
        word = self.pauli_word(letters + "I" * (N_WIRES - len(letters)))
        expected_expval = (final_state.conj() @ word @ final_state).real.item()
        expected_probabilities = torch.diagonal(self.reduced_density_matrix(final_state, kept_wires)).real.numpy()
        expval, probabilities = follow_up()
        np.testing.assert_allclose(expval, expected_expval, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(
            probabilities, expected_probabilities, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_odd_observables_need_the_displacement_vector(self, prepared_device, state):
        reduced_state = prepared_device.reduced_state([0, 1])

        def expval(displacement_vector):
            @qml.qnode(qml.device("nif.qubit", wires=2))
            def circuit():
                GaussianStatePrep(
                    reduced_state.covariance_matrix,
                    wires=[0, 1],
                    validate=False,
                    displacement_vector=displacement_vector,
                )
                return qml.expval(qml.X(0))

            return float(circuit())

        expected = (state.conj() @ self.pauli_word("X" + "I" * (N_WIRES - 1)) @ state).real.item()
        assert abs(expected) > ATOL_MATRIX_COMPARISON
        np.testing.assert_allclose(
            expval(reduced_state.displacement_vector),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(expval(None), 0.0, atol=ATOL_MATRIX_COMPARISON)

    def test_reduced_state_embedded_in_a_larger_device(self, prepared_device, state):
        reduced_state = prepared_device.reduced_state([0, 1, 2])
        density_matrix = self.reduced_density_matrix(state, [0, 1, 2])

        @qml.qnode(qml.device("nif.qubit", wires=5))
        def circuit():
            GaussianStatePrep(
                reduced_state.covariance_matrix,
                wires=[1, 2, 3],
                displacement_vector=reduced_state.displacement_vector,
            )
            return (
                qml.expval(qml.X(1)),
                qml.expval(qml.Y(2)),
                qml.expval(qml.X(1) @ qml.Y(2) @ qml.X(3)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
            )

        expected = [
            torch.trace(density_matrix @ self.pauli_word(letters)).real.item()
            for letters in ("XII", "IYI", "XYX", "ZII")
        ]
        np.testing.assert_allclose(
            [float(value) for value in circuit()], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_sampling_from_the_reduced_state(self, prepared_device, state):
        reduced_state = prepared_device.reduced_state([0, 1])
        density_matrix = self.reduced_density_matrix(state, [0, 1])

        @qml.set_shots(100_000)
        @qml.qnode(qml.device("nif.qubit", wires=2))
        def circuit():
            GaussianStatePrep(
                reduced_state.covariance_matrix,
                wires=[0, 1],
                displacement_vector=reduced_state.displacement_vector,
            )
            return qml.expval(qml.X(0)), qml.expval(qml.Y(1))

        expected = [torch.trace(density_matrix @ self.pauli_word(letters)).real.item() for letters in ("XI", "IY")]
        np.testing.assert_allclose([float(value) for value in circuit()], expected, atol=ATOL_APPROX_COMPARISON)

    def test_reduced_state_of_a_reduced_state(self, prepared_device, operations):
        first = prepared_device.reduced_state([0, 1, 2, 3])
        device = mc.NonInteractingFermionicDevice(wires=4)
        device.apply([first, *self.brick_wall(device.wires, depth=2, seed=TEST_SEED + 1)])
        second = device.reduced_state([0, 1])
        reference_device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        reference_device.apply(
            [prepared_device.state_prep_op, *operations, *self.brick_wall(range(4), depth=2, seed=TEST_SEED + 1)]
        )
        reference = reference_device.reduced_state([0, 1])
        assert second.num_params == 2
        assert np.abs(qml.math.toarray(second.displacement_vector)).max() > ATOL_APPROX_COMPARISON
        for value, expected in zip(second.data, reference.data):
            np.testing.assert_allclose(
                qml.math.toarray(value),
                qml.math.toarray(expected),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("measurement", [qml.vn_entropy, qml.purity, qml.density_matrix])
    def test_measurements_of_a_displaced_gaussian_state_are_rejected(self, prepared_device, measurement):
        reduced_state = prepared_device.reduced_state([0, 1, 2])

        @qml.qnode(mc.NonInteractingFermionicDevice(wires=3))
        def circuit():
            qml.apply(reduced_state)
            return measurement(wires=[0, 1])

        with pytest.raises(NotImplementedError, match="without displacement"):
            circuit()

    @pytest.mark.parametrize("magnitude", [0.0, 1e-13])
    def test_gaussian_state_with_a_vanishing_displacement_is_not_displaced(self, magnitude):
        covariance = PlusState(wires=[0, 1, 2]).covariance_matrix
        op = GaussianStatePrep(covariance, wires=[0, 1, 2], validate=False, displacement_vector=np.full(6, magnitude))
        device = mc.NonInteractingFermionicDevice(wires=3)
        device.apply([op])
        assert device.reduced_state([1, 2]).num_params == 1

        @qml.qnode(mc.NonInteractingFermionicDevice(wires=3))
        def circuit():
            qml.apply(op)
            return qml.purity(wires=[0, 1])

        assert np.isfinite(float(circuit()))

    def test_reporter_snippet_with_the_kept_wires_first(self):
        n_wires = 4
        device = mc.NIFDevice(wires=n_wires)
        angles = self.random_angles(n_wires)
        amplitudes = self.amplitudes_from_angles(angles)
        device.apply(ProductState(state=amplitudes, wires=device.wires))
        reduced_state = device.partial_trace([2, 3])
        expected = ProductState(amplitudes[:2], wires=[0, 1])
        np.testing.assert_allclose(
            qml.math.toarray(reduced_state.covariance_matrix),
            qml.math.toarray(expected.covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            qml.math.toarray(reduced_state.displacement_vector),
            qml.math.toarray(expected.displacement_vector),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            qml.math.toarray(device.reduced_covariance_matrix([2, 3])),
            qml.math.toarray(ProductState(amplitudes[2:], wires=[0, 1]).covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("kept_wires", [[0, 1], [2, 3], [1, 2, 3], [3, 4]])
    def test_states_without_displacement_are_reduced_as_before(self, operations, kept_wires):
        basis_state = np.array([1, 0, 1, 1, 0])
        state_preparations = [
            qml.BasisState(basis_state, wires=range(N_WIRES)),
            ProductState.from_basis_state(basis_state, wires=range(N_WIRES)),
            GaussianStatePrep(
                ProductState.from_basis_state(basis_state, wires=range(N_WIRES)).covariance_matrix,
                wires=range(N_WIRES),
            ),
        ]
        amplitudes = torch.eye(2, dtype=torch.complex128)[torch.as_tensor(basis_state)]
        density_matrix = self.reduced_density_matrix(self.state_vector(amplitudes, operations), kept_wires)
        expected = self.covariance_of(
            lambda operator: torch.trace(density_matrix @ operator), self.majoranas(len(kept_wires))
        )
        for state_preparation in state_preparations:
            device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
            device.apply([state_preparation] + operations)
            reduced_state = device.reduced_state(kept_wires)
            assert reduced_state.num_params == 1
            assert reduced_state.ndim_params == (2,)
            np.testing.assert_allclose(
                qml.math.toarray(reduced_state.covariance_matrix),
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
            np.testing.assert_allclose(qml.math.toarray(reduced_state.displacement_vector), 0.0, atol=0.0)

    @pytest.mark.parametrize("r_dtype, c_dtype", [(torch.float32, torch.complex64), (torch.float64, torch.complex128)])
    def test_displacement_vector_dtype(self, angles, operations, r_dtype, c_dtype):
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES, r_dtype=r_dtype, c_dtype=c_dtype)
        device.apply([ProductState(self.amplitudes_from_angles(angles), wires=range(N_WIRES))] + operations)
        assert device.displacement_vector.dtype == r_dtype
        reduced_state = device.reduced_state([0, 1])
        assert reduced_state.displacement_vector.dtype == reduced_state.covariance_matrix.dtype

    def test_batched_product_state(self, operations):
        batch_size = 3
        kept_wires = [0, 1, 2]
        angles = self.random_angles(batch_size=batch_size)
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply([ProductState(self.amplitudes_from_angles(angles), wires=range(N_WIRES))] + operations)
        reduced_state = device.reduced_state(kept_wires)
        assert tuple(qml.math.shape(reduced_state.covariance_matrix)) == (batch_size, 6, 6)
        assert tuple(qml.math.shape(reduced_state.displacement_vector)) == (batch_size, 6)
        assert reduced_state.batch_size == batch_size
        follow_up_operations = self.brick_wall(kept_wires, 1, TEST_SEED + 3)

        @qml.qnode(qml.device("nif.qubit", wires=3))
        def follow_up():
            GaussianStatePrep(
                reduced_state.covariance_matrix,
                wires=kept_wires,
                displacement_vector=reduced_state.displacement_vector,
            )
            for operation in follow_up_operations:
                qml.apply(operation)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0) @ qml.Z(1)), qml.expval(qml.Z(1) @ qml.Z(2))

        results = [qml.math.toarray(result) for result in follow_up()]
        for index in range(batch_size):
            final_state = self.state_vector(
                self.amplitudes_from_angles(angles[index]), operations + follow_up_operations
            )
            expected = [
                (final_state.conj() @ self.pauli_word(letters) @ final_state).real.item()
                for letters in ("XIIII", "YZIII", "IZZII")
            ]
            np.testing.assert_allclose(
                [result[index] for result in results],
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("depth", [0, 2])
    def test_gradient_through_the_reduced_state(self, depth):
        operations = self.brick_wall(range(N_WIRES), depth, TEST_SEED)
        kept_wires = [0, 1, 2]
        follow_up_operations = self.brick_wall(kept_wires, 1, TEST_SEED + 4)
        word = self.pauli_word("YZ" + "I" * (N_WIRES - 2))

        def reduced_expval(angles):
            device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
            device.apply([ProductState(self.amplitudes_from_angles(angles), wires=range(N_WIRES))] + operations)
            reduced_state = device.reduced_state(kept_wires)

            @qml.qnode(qml.device("nif.qubit", wires=3), interface="torch", diff_method="backprop")
            def follow_up(covariance_matrix, displacement_vector):
                GaussianStatePrep(covariance_matrix, wires=kept_wires, displacement_vector=displacement_vector)
                for operation in follow_up_operations:
                    qml.apply(operation)
                return qml.expval(qml.Y(0) @ qml.Z(1))

            return follow_up(reduced_state.covariance_matrix, reduced_state.displacement_vector)

        def dense_expval(angles):
            final_state = self.state_vector(self.amplitudes_from_angles(angles), operations + follow_up_operations)
            return (final_state.conj() @ word @ final_state).real

        angles = self.random_angles().requires_grad_(True)
        (gradient,) = torch.autograd.grad(reduced_expval(angles), angles)
        (expected_gradient,) = torch.autograd.grad(dense_expval(angles), angles)
        assert expected_gradient.abs().max() > ATOL_MATRIX_COMPARISON
        torch.testing.assert_close(
            gradient.to(expected_gradient.dtype),
            expected_gradient,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
