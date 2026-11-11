import itertools

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.sampling_strategies import ProductBasisSampling
from matchcake.operations.state_preparation import ProductState

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_SHOTS = 20_000
PAULI_MATRICES = [qml.matrix(qml.X(0)), qml.matrix(qml.Y(0)), qml.matrix(qml.Z(0))]


class TestProductBasisSampling:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @staticmethod
    def random_amplitudes(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
        amplitudes = rng.normal(size=(n_qubits, 2)) + 1j * rng.normal(size=(n_qubits, 2))
        return amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)

    @staticmethod
    def basis_amplitudes(bits) -> np.ndarray:
        amplitudes = np.zeros((len(bits), 2), dtype=complex)
        amplitudes[np.arange(len(bits)), bits] = 1.0
        return amplitudes

    @staticmethod
    def random_operations(n_qubits: int, rng: np.random.Generator) -> list:
        operations = []
        for _ in range(2):
            for wire in range(n_qubits - 1):
                angles = rng.normal(size=3)
                operations += [
                    qml.IsingXX(angles[0], wires=[wire, wire + 1]),
                    qml.IsingYY(angles[1], wires=[wire, wire + 1]),
                    qml.SingleExcitation(angles[2], wires=[wire, wire + 1]),
                ]
        return operations

    @staticmethod
    def extended_covariance_matrix(amplitudes: np.ndarray, operations: list) -> torch.Tensor:
        device = NonInteractingFermionicDevice(wires=len(amplitudes))
        device.apply([ProductState(amplitudes, wires=range(len(amplitudes)))] + operations)
        return device.extended_covariance_matrix

    @staticmethod
    def reference_probabilities(amplitudes: np.ndarray, operations: list, bloch_vectors: np.ndarray) -> np.ndarray:
        n_qubits = len(amplitudes)

        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def state_circuit():
            ProductState(amplitudes, wires=range(n_qubits))
            for operation in operations:
                qml.apply(operation)
            return qml.state()

        state = np.asarray(state_circuit())
        bloch_vectors = bloch_vectors / np.linalg.norm(bloch_vectors, axis=-1, keepdims=True)
        probabilities = []
        for outcome in itertools.product([0, 1], repeat=n_qubits):
            projector = np.ones((1, 1))
            for bit, bloch_vector in zip(outcome, bloch_vectors):
                observable = sum(component * pauli for component, pauli in zip(bloch_vector, PAULI_MATRICES))
                projector = np.kron(projector, (np.eye(2) + (1 - 2 * bit) * observable) / 2)
            probabilities.append(np.real(state.conj() @ projector @ state))
        return np.array(probabilities)

    @staticmethod
    def all_outcomes(n_qubits: int) -> np.ndarray:
        return np.array(list(itertools.product([0, 1], repeat=n_qubits)))

    @pytest.mark.parametrize("input_kind", ["even", "odd", "product"])
    @pytest.mark.parametrize("bases", ["XXXX", "YZXY", "ZZZZ", "random"])
    def test_outcome_probabilities_match_default_qubit(self, input_kind, bases):
        rng = np.random.default_rng(TEST_SEED)
        n_qubits = 4
        amplitudes = {
            "even": self.basis_amplitudes([0, 1, 1, 0]),
            "odd": self.basis_amplitudes([0, 1, 0, 0]),
            "product": self.random_amplitudes(n_qubits, rng),
        }[input_kind]
        operations = self.random_operations(n_qubits, rng)
        if bases == "random":
            bloch_vectors = rng.normal(size=(n_qubits, 3))
        else:
            bloch_vectors = ProductBasisSampling.pauli_bloch_vectors(list(bases))
        strategy = ProductBasisSampling()
        probabilities = strategy.outcome_probabilities(
            self.extended_covariance_matrix(amplitudes, operations), bloch_vectors, self.all_outcomes(n_qubits)
        )
        expected = self.reference_probabilities(amplitudes, operations, bloch_vectors)
        np.testing.assert_allclose(probabilities.numpy(), expected, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("input_kind", ["even", "odd", "product"])
    def test_x_basis_samples_match_default_qubit_with_hadamard_layer(self, input_kind):
        set_seed(TEST_SEED)
        rng = np.random.default_rng(TEST_SEED)
        n_qubits = 3
        amplitudes = {
            "even": self.basis_amplitudes([1, 1, 0]),
            "odd": self.basis_amplitudes([0, 0, 1]),
            "product": self.random_amplitudes(n_qubits, rng),
        }[input_kind]
        operations = self.random_operations(n_qubits, rng)

        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def hadamard_layer_circuit():
            ProductState(amplitudes, wires=range(n_qubits))
            for operation in operations:
                qml.apply(operation)
            for wire in range(n_qubits):
                qml.Hadamard(wires=wire)
            return qml.probs()

        expected = np.asarray(hadamard_layer_circuit())
        samples = ProductBasisSampling().sample(
            self.extended_covariance_matrix(amplitudes, operations),
            ProductBasisSampling.pauli_bloch_vectors(["X"] * n_qubits),
            N_SHOTS,
        )
        indices = samples @ (1 << np.arange(n_qubits)[::-1])
        empirical = np.bincount(indices, minlength=2**n_qubits) / N_SHOTS
        np.testing.assert_allclose(empirical, expected, atol=ATOL_APPROX_COMPARISON)

    def test_computational_basis_matches_device_probabilities(self):
        rng = np.random.default_rng(TEST_SEED)
        n_qubits = 4
        amplitudes = self.random_amplitudes(n_qubits, rng)
        device = NonInteractingFermionicDevice(wires=n_qubits)
        device.apply([ProductState(amplitudes, wires=range(n_qubits))] + self.random_operations(n_qubits, rng))
        probabilities = ProductBasisSampling().outcome_probabilities(
            device.extended_covariance_matrix,
            ProductBasisSampling.pauli_bloch_vectors(["Z"] * n_qubits),
            self.all_outcomes(n_qubits),
        )
        np.testing.assert_allclose(
            probabilities.numpy(), device.analytic_probability().numpy(), atol=ATOL_SCALAR_COMPARISON
        )

    def test_impossible_outcome_has_zero_probability(self):
        extended_covariance_matrix = self.extended_covariance_matrix(self.basis_amplitudes([0, 1, 1]), [])
        probabilities = ProductBasisSampling().outcome_probabilities(
            extended_covariance_matrix,
            ProductBasisSampling.pauli_bloch_vectors(["Z", "Z", "X"]),
            np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        )
        np.testing.assert_allclose(probabilities.numpy(), [0.5, 0.0, 0.0], atol=ATOL_SCALAR_COMPARISON)

    def test_outcome_probabilities_with_batched_covariance(self):
        rng = np.random.default_rng(TEST_SEED)
        n_qubits = 3
        bloch_vectors = rng.normal(size=(n_qubits, 3))
        outcomes = self.all_outcomes(n_qubits)
        batch = [
            self.extended_covariance_matrix(
                self.random_amplitudes(n_qubits, rng), self.random_operations(n_qubits, rng)
            )
            for _ in range(2)
        ]
        strategy = ProductBasisSampling()
        probabilities = strategy.outcome_probabilities(torch.stack(batch), bloch_vectors, outcomes)
        assert tuple(probabilities.shape) == (len(outcomes), 2)
        for batch_index, extended_covariance_matrix in enumerate(batch):
            np.testing.assert_allclose(
                probabilities[:, batch_index].numpy(),
                strategy.outcome_probabilities(extended_covariance_matrix, bloch_vectors, outcomes).numpy(),
                atol=ATOL_SCALAR_COMPARISON,
            )

    def test_outcome_probabilities_keeps_numpy_backend(self):
        extended_covariance_matrix = self.extended_covariance_matrix(self.basis_amplitudes([0, 1]), []).numpy()
        probabilities = ProductBasisSampling().outcome_probabilities(
            extended_covariance_matrix, ProductBasisSampling.pauli_bloch_vectors(["X", "Z"]), np.array([[0, 1]])
        )
        assert isinstance(probabilities, np.ndarray)
        np.testing.assert_allclose(probabilities, [0.5], atol=ATOL_SCALAR_COMPARISON)

    def test_sample_shape_with_batch_and_chunks(self):
        rng = np.random.default_rng(TEST_SEED)
        n_qubits = 3
        extended_covariance_matrix = self.extended_covariance_matrix(
            self.random_amplitudes(n_qubits, rng), self.random_operations(n_qubits, rng)
        )
        samples = ProductBasisSampling(shots_chunk_size=4).sample(
            torch.stack([extended_covariance_matrix] * 2),
            ProductBasisSampling.pauli_bloch_vectors(["X", "Y", "Z"]),
            10,
            show_progress=False,
        )
        assert samples.shape == (10, 2, n_qubits)
        assert samples.dtype == np.int64
        assert set(np.unique(samples)) <= {0, 1}

    def test_sample_without_shots(self):
        extended_covariance_matrix = self.extended_covariance_matrix(self.basis_amplitudes([0, 1]), [])
        samples = ProductBasisSampling().sample(
            extended_covariance_matrix, ProductBasisSampling.pauli_bloch_vectors(["X", "X"]), 0
        )
        assert samples.shape == (0, 2)

    def test_sample_deterministic_basis_state(self):
        extended_covariance_matrix = self.extended_covariance_matrix(self.basis_amplitudes([1, 0, 1]), [])
        samples = ProductBasisSampling().sample(
            extended_covariance_matrix, ProductBasisSampling.pauli_bloch_vectors(["Z", "Z", "Z"]), 50
        )
        np.testing.assert_array_equal(samples, np.broadcast_to([1, 0, 1], (50, 3)))

    def test_default_chunk_size(self):
        assert ProductBasisSampling().shots_chunk_size == ProductBasisSampling.DEFAULT_SHOTS_CHUNK_SIZE
        assert ProductBasisSampling(shots_chunk_size=3).shots_chunk_size == 3

    @pytest.mark.parametrize("use_batch_method", [True, False])
    def test_generate_samples_from_device(self, use_batch_method):
        set_seed(TEST_SEED)
        amplitudes = self.basis_amplitudes([0, 1])
        device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        device.apply([ProductState(amplitudes, wires=range(2)), qml.SingleExcitation(0.8, wires=[0, 1])])
        strategy = ProductBasisSampling()
        generate = strategy.batch_generate_samples if use_batch_method else strategy.generate_samples
        z_samples = generate(device, None)
        np.testing.assert_allclose(
            np.mean(z_samples, axis=0), [np.sin(0.4) ** 2, np.cos(0.4) ** 2], atol=ATOL_APPROX_COMPARISON
        )
        x_samples = generate(device, None, bloch_vectors=ProductBasisSampling.pauli_bloch_vectors(["X", "X"]))
        x_eigenvalues = 1 - 2 * x_samples
        np.testing.assert_allclose(
            np.mean(x_eigenvalues[:, 0] * x_eigenvalues[:, 1]), np.sin(0.8), atol=ATOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    @pytest.mark.parametrize(
        "bloch_vector",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.3, -0.5, 0.81]],
    )
    def test_measurement_frames_are_proper_rotations(self, bloch_vector, sign):
        bloch_vector = torch.tensor(bloch_vector, dtype=torch.float64)
        bloch_vector = bloch_vector / torch.linalg.norm(bloch_vector)
        frame = ProductBasisSampling.measurement_frames(bloch_vector, torch.tensor(sign, dtype=torch.float64))
        np.testing.assert_allclose((frame @ frame.T).numpy(), np.eye(3), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(torch.linalg.det(frame).item(), 1.0, atol=ATOL_SCALAR_COMPARISON)
        expected_w = [bloch_vector[2], -sign * bloch_vector[1], sign * bloch_vector[0]]
        np.testing.assert_allclose(frame[0].numpy(), expected_w, atol=ATOL_MATRIX_COMPARISON)

    def test_measurement_frames_broadcast_over_signs(self):
        frames = ProductBasisSampling.measurement_frames(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), torch.tensor([1.0, -1.0], dtype=torch.float64)
        )
        assert tuple(frames.shape) == (2, 3, 3)
        np.testing.assert_allclose(frames[0, 0].numpy(), [0.0, 0.0, 1.0], atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(frames[1, 0].numpy(), [0.0, 0.0, -1.0], atol=ATOL_MATRIX_COMPARISON)

    def test_pauli_bloch_vectors(self):
        np.testing.assert_allclose(
            ProductBasisSampling.pauli_bloch_vectors(["X", "Y", "Z"]), np.eye(3), atol=ATOL_MATRIX_COMPARISON
        )
        assert ProductBasisSampling.pauli_bloch_vectors([]).shape == (0, 3)

    def test_pauli_bloch_vectors_unknown_letter_raises(self):
        with pytest.raises(ValueError, match="Unknown Pauli bases"):
            ProductBasisSampling.pauli_bloch_vectors(["X", "W"])

    @pytest.mark.parametrize(
        "bloch_vectors, match",
        [
            (np.ones((2, 2)), "must have shape \\(n, 3\\)"),
            (np.ones(3), "must have shape \\(n, 3\\)"),
            (np.ones((3, 3)), "extended covariance matrix must have shape"),
            (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "must be non-zero"),
        ],
    )
    def test_invalid_inputs_raise(self, bloch_vectors, match):
        extended_covariance_matrix = self.extended_covariance_matrix(self.basis_amplitudes([0, 1]), [])
        with pytest.raises(ValueError, match=match):
            ProductBasisSampling().sample(extended_covariance_matrix, bloch_vectors, 1)

    def test_outcomes_with_wrong_number_of_qubits_raise(self):
        extended_covariance_matrix = self.extended_covariance_matrix(self.basis_amplitudes([0, 1]), [])
        with pytest.raises(ValueError, match="Expected outcomes on 2 qubits"):
            ProductBasisSampling().outcome_probabilities(
                extended_covariance_matrix, ProductBasisSampling.pauli_bloch_vectors(["X", "X"]), np.array([[0, 1, 0]])
            )
