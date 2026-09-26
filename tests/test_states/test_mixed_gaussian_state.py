import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import BatchHamiltonian, MixedGaussianState, NonInteractingFermionicDevice
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.operations.state_preparation import ProductState
from matchcake.utils.covariance import block_diagonal_covariance
from matchcake.utils.majorana import get_majorana

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)

N_SHOTS = 20_000


class TestMixedGaussianState:
    @staticmethod
    def brick_wall(wires, depth: int, rng: np.random.Generator):
        """Random layers of ``CompRxRx``, ``CompRyRy``, ``CompRzRz`` and an ``fSWAP`` on consecutive wires.

        :param wires: The wires of the circuit.
        :type wires: Iterable[int]
        :param depth: Number of layers.
        :type depth: int
        :param rng: Random generator drawing the angles.
        :type rng: np.random.Generator
        :return: The operations.
        :rtype: List[Operation]
        """
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
    def dense_state(basis_state, operations, n_wires: int) -> np.ndarray:
        """Run the circuit on ``default.qubit`` and return the state vector.

        :param basis_state: Initial computational-basis state.
        :type basis_state: np.ndarray
        :param operations: Operations of the circuit, built outside of any queuing context.
        :type operations: List[Operation]
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: The state vector.
        :rtype: np.ndarray
        """
        device = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(device)
        def circuit():
            qml.BasisState(np.asarray(basis_state), wires=range(n_wires))
            for operation in operations:
                qml.apply(operation)
            return qml.state()

        return np.asarray(circuit())

    @staticmethod
    def dense_covariance(density_matrix: np.ndarray, n_wires: int) -> np.ndarray:
        """Brute-force Majorana covariance matrix ``i Tr[rho c_mu c_nu]`` of a density matrix.

        :param density_matrix: Density matrix of shape ``(2^n, 2^n)``.
        :type density_matrix: np.ndarray
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: Real antisymmetric matrix of shape ``(2n, 2n)``.
        :rtype: np.ndarray
        """
        covariance = np.zeros((2 * n_wires, 2 * n_wires), dtype=complex)
        for mu in range(2 * n_wires):
            for nu in range(2 * n_wires):
                if mu != nu:
                    covariance[mu, nu] = 1j * np.trace(
                        density_matrix @ get_majorana(mu, n_wires) @ get_majorana(nu, n_wires)
                    )
        return covariance.real

    @staticmethod
    def dense_unitary(operations, n_wires: int) -> np.ndarray:
        """Dense unitary of a circuit on ``range(n_wires)``.

        :param operations: Operations of the circuit.
        :type operations: List[Operation]
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: Unitary matrix of shape ``(2^n, 2^n)``.
        :rtype: np.ndarray
        """
        return np.asarray(qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(n_wires)))

    @pytest.fixture
    def reduced_case(self):
        """A 5-wire pure state reduced to the contiguous block ``[1, 2, 3]`` with its dense reference."""
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 5
        basis_state = np.array([1, 0, 1, 1, 0])
        operations = self.brick_wall(range(n_wires), 2, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([qml.BasisState(basis_state, wires=range(n_wires))] + operations)
        state = device.partial_trace([0, 4])
        psi = self.dense_state(basis_state, operations, n_wires)
        reduced_density = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [0, 4]))
        follow_up = self.brick_wall(range(3), 1, rng)
        unitary = self.dense_unitary(follow_up, 3)
        evolved_density = unitary @ reduced_density @ unitary.conj().T
        return state, reduced_density, follow_up, evolved_density

    def test_init_rejects_odd_or_non_square_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            MixedGaussianState(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="shape"):
            MixedGaussianState(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="shape"):
            MixedGaussianState(np.zeros((2, 2, 4, 4)))

    def test_init_rejects_non_antisymmetric_matrix(self):
        with pytest.raises(ValueError, match="antisymmetric"):
            MixedGaussianState(np.eye(4))

    def test_init_rejects_wrong_number_of_wires(self):
        with pytest.raises(ValueError, match="Expected 2 wires"):
            MixedGaussianState(np.zeros((4, 4)), wires=[0, 1, 2])

    def test_init_rejects_non_consecutive_wires(self):
        with pytest.raises(ValueError, match="consecutive"):
            MixedGaussianState(np.zeros((4, 4)), wires=[0, 2])
        with pytest.raises(ValueError, match="consecutive"):
            MixedGaussianState(np.zeros((4, 4)), wires=["a", "b"])

    def test_init_accepts_shifted_consecutive_wires(self):
        state = MixedGaussianState(np.zeros((4, 4)), wires=[3, 4])
        assert state.wires.tolist() == [3, 4]
        assert state.source_wires.tolist() == [3, 4]

    def test_defaults(self):
        state = MixedGaussianState(np.zeros((6, 6)))
        assert state.wires.tolist() == [0, 1, 2]
        assert state.num_wires == 3
        assert state.batch_shape == ()
        assert state.atol == MixedGaussianState.DEFAULT_ATOL
        assert state.device_kwargs == {}

    def test_repr(self):
        state = MixedGaussianState.from_polarizations([1.0, 0.0])
        assert repr(state) == "MixedGaussianState(num_wires=2, batch_shape=(), num_mixed_modes=1, num_pure_states=2)"

    def test_majorana_indices(self):
        np.testing.assert_array_equal(MixedGaussianState.majorana_indices([0, 2]), [0, 1, 4, 5])
        np.testing.assert_array_equal(MixedGaussianState.majorana_indices([]), np.zeros(0, dtype=int))

    def test_from_polarizations_identity_sptm(self):
        polarizations = np.array([0.5, -1.0, 0.0])
        state = MixedGaussianState.from_polarizations(polarizations)
        np.testing.assert_allclose(
            state.covariance_matrix,
            block_diagonal_covariance(polarizations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            np.sort(np.abs(state.mode_polarizations)),
            [0.0, 0.5, 1.0],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_from_polarizations_with_sptm(self):
        polarizations = np.array([0.5, -0.2])
        sptm = SingleParticleTransitionMatrixOperation.random_params(wires=[0, 1], seed=TEST_SEED).real
        state = MixedGaussianState.from_polarizations(polarizations, sptm=sptm, atol=1e-6)
        expected = sptm.T @ block_diagonal_covariance(polarizations) @ sptm
        np.testing.assert_allclose(
            state.covariance_matrix, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert state.atol == 1e-6

    def test_pure_state_has_single_member(self):
        state = MixedGaussianState.from_polarizations([1.0, -1.0, 1.0])
        assert state.is_pure
        assert state.mixed_modes == []
        assert state.num_mixed_modes == 0
        assert state.num_pure_states == 1
        assert state.basis_states.shape == (1, 3)
        np.testing.assert_allclose(state.weights, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        # The normal modes may be permuted by the decomposition, but the state is the basis state |010>.
        expected_probs = np.zeros(8)
        expected_probs[0b010] = 1.0
        np.testing.assert_allclose(state.probs(), expected_probs, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(state.purity, 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(state.entropy(), 0.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_mixed_modes_basis_states_and_weights(self):
        state = MixedGaussianState.from_polarizations([0.6, -1.0, -0.2])
        assert not state.is_pure
        assert state.num_mixed_modes == 2
        assert state.num_pure_states == 4
        assert state.basis_states.shape == (4, 3)
        # Every member is a distinct basis state of the normal modes with the product weights.
        assert len({tuple(bits) for bits in state.basis_states}) == 4
        expected_weights = [0.8 * 0.4, 0.8 * 0.6, 0.2 * 0.4, 0.2 * 0.6]
        np.testing.assert_allclose(
            np.sort(state.weights), np.sort(expected_weights), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        # The covariance matrix is already block diagonal, so the state is diagonal in the computational basis.
        bits = np.array([[(index >> shift) & 1 for shift in (2, 1, 0)] for index in range(8)])
        polarizations = np.array([0.6, -1.0, -0.2])
        expected_probs = np.prod(0.5 * (1.0 + (1 - 2 * bits) * polarizations), axis=-1)
        np.testing.assert_allclose(
            state.probs(), expected_probs, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            state.purity, (1 + 0.36) / 2 * (1 + 0.04) / 2, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_atol_controls_purity_detection(self):
        polarizations = [1.0 - 1e-5, 0.3]
        assert MixedGaussianState.from_polarizations(polarizations, atol=1e-8).num_mixed_modes == 2
        assert MixedGaussianState.from_polarizations(polarizations, atol=1e-4).num_mixed_modes == 1

    def test_entropy_matches_binary_entropy_and_base(self):
        state = MixedGaussianState.from_polarizations([0.0, 1.0, 0.5])
        expected = np.log(2) + (-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
        np.testing.assert_allclose(state.entropy(), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(
            state.entropy(base=2), expected / np.log(2), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_batched_polarizations_union_of_mixed_modes(self):
        polarizations = np.array([[1.0, 0.5], [-0.3, 1.0]])
        state = MixedGaussianState.from_polarizations(polarizations)
        assert state.batch_shape == (2,)
        # Each element has one mixed mode; the union over the batch decides the members.
        assert state.num_mixed_modes == 1
        assert state.basis_states.shape == (2, 2, 2)
        np.testing.assert_allclose(np.sum(state.weights, axis=-1), [1.0, 1.0], atol=ATOL_SCALAR_COMPARISON)
        np.testing.assert_array_equal(state.is_pure, [False, False])
        assert state.entropy().shape == (2,)
        assert state.purity.shape == (2,)

    def test_sptm_operation_and_pure_state_operations(self):
        state = MixedGaussianState.from_polarizations([0.5, 1.0])
        operation = state.sptm_operation
        assert isinstance(operation, SingleParticleTransitionMatrixOperation)
        assert operation.wires.tolist() == [0, 1]
        np.testing.assert_allclose(
            operation.matrix(), state.sptm, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        prep, sptm_op = state.pure_state_operations(1)
        assert isinstance(prep, ProductState)
        np.testing.assert_array_equal(prep.basis_state_bits(), state.basis_states[1])
        assert isinstance(sptm_op, SingleParticleTransitionMatrixOperation)

    def test_pure_states_iterates_members(self):
        state = MixedGaussianState.from_polarizations([0.5, 1.0, -0.1])
        members = list(state.pure_states())
        assert len(members) == state.num_pure_states == 4
        weights = np.array([float(weight) for weight, _, _ in members])
        np.testing.assert_allclose(weights, state.weights, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        for index, (_, prep, sptm_op) in enumerate(members):
            np.testing.assert_array_equal(prep.basis_state_bits(), state.basis_states[index])
            assert sptm_op is members[0][2]

    def test_ensemble_operations_unbatched(self):
        state = MixedGaussianState.from_polarizations([0.5, 1.0, -0.1])
        prep, sptm_op = state.ensemble_operations()
        assert prep.batch_size == 4
        np.testing.assert_array_equal(prep.basis_state_bits(), state.basis_states)
        assert sptm_op.batch_size is None

    def test_ensemble_operations_batched_repeats_sptm(self):
        polarizations = np.array([[1.0, 0.5], [-0.3, 1.0]])
        state = MixedGaussianState.from_polarizations(polarizations)
        num_members = state.num_pure_states
        prep, sptm_op = state.ensemble_operations()
        assert prep.batch_size == 2 * num_members
        assert sptm_op.batch_size == 2 * num_members
        sptm = np.asarray(sptm_op.matrix())
        for batch_index in range(2):
            for member in range(num_members):
                np.testing.assert_allclose(
                    sptm[batch_index * num_members + member], state.sptm[batch_index], atol=ATOL_MATRIX_COMPARISON
                )

    def test_ensemble_reconstructs_density_matrix(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        n_wires = state.num_wires
        reconstructed = np.zeros((2**n_wires, 2**n_wires), dtype=complex)
        for weight, prep, sptm_op in state.pure_states():
            unitary = np.asarray(
                SingleParticleTransitionMatrixOperation.to_unitary_matrix(torch.as_tensor(sptm_op.matrix()))
            )
            vector = unitary @ prep.state_vector().reshape(-1)
            reconstructed += float(weight) * np.outer(vector, vector.conj())
        np.testing.assert_allclose(
            reconstructed, reduced_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_density_matrix_matches_qubit_partial_trace(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        np.testing.assert_allclose(
            state.density_matrix(), reduced_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_density_matrix_batched_and_backend(self):
        polarizations = torch.tensor([[1.0, 0.0], [-1.0, 0.4]], dtype=torch.float64)
        state = MixedGaussianState.from_polarizations(polarizations)
        density = state.density_matrix()
        assert isinstance(density, torch.Tensor)
        assert density.shape == (2, 4, 4)
        expected_first = np.diag([0.5, 0.5, 0.0, 0.0])
        expected_second = np.diag([0.0, 0.0, 0.7, 0.3])
        np.testing.assert_allclose(density[0].numpy(), expected_first, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(density[1].numpy(), expected_second, atol=ATOL_MATRIX_COMPARISON)

    def test_entropy_and_purity_match_dense(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        eigenvalues = np.linalg.eigvalsh(reduced_density)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        np.testing.assert_allclose(
            state.entropy(),
            -np.sum(eigenvalues * np.log(eigenvalues)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            state.purity,
            np.trace(reduced_density @ reduced_density).real,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_covariance_matrix_matches_dense(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        np.testing.assert_allclose(
            state.covariance_matrix,
            self.dense_covariance(reduced_density, state.num_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_extended_covariance_matrix_has_zero_displacement(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        extended = state.extended_covariance_matrix
        assert extended.shape == (5, 5)
        np.testing.assert_allclose(extended[:4, :4], state.covariance_matrix, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(extended[4, :], 0.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(extended[:, 4], 0.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_expval_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        observable = qml.PauliX(0) @ qml.PauliY(1) + 0.5 * qml.PauliZ(2) + 0.3 * qml.PauliX(1) @ qml.PauliX(2)
        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
        value = state.execute(follow_up, observable, "expval", mode=mode)
        np.testing.assert_allclose(value, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_batch_hamiltonian_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        terms = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(1) @ qml.PauliX(2)]
        coefficients = [0.2, -0.7]
        observable = BatchHamiltonian(coefficients, terms)
        expected = [
            coefficient * np.trace(evolved_density @ qml.matrix(term, wire_order=range(3))).real
            for coefficient, term in zip(coefficients, terms)
        ]
        value = state.execute(follow_up, observable, "expval", mode=mode)
        np.testing.assert_allclose(value, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_probs_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        probabilities = state.execute(follow_up, output_type="probs", mode=mode)
        np.testing.assert_allclose(
            probabilities, np.diag(evolved_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_marginal_probs_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        marginal = np.asarray(qml.math.partial_trace(evolved_density, [1]))
        probabilities = state.execute(follow_up, output_type="probs", mode=mode, wires=[0, 2])
        np.testing.assert_allclose(
            probabilities, np.diag(marginal).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_samples_follow_probabilities(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        np.random.seed(TEST_SEED)
        samples = state.execute(follow_up, output_type="samples", mode=mode, shots=N_SHOTS)
        assert samples.shape == (N_SHOTS, 3)
        indices = samples @ (2 ** np.arange(3)[::-1])
        frequencies = np.bincount(indices, minlength=8) / N_SHOTS
        np.testing.assert_allclose(frequencies, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_shots_expval_and_probs_are_estimated_from_samples(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        np.random.seed(TEST_SEED)
        observable = qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2)
        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
        value = state.execute(follow_up, observable, "expval", mode=mode, shots=N_SHOTS)
        np.testing.assert_allclose(value, expected, atol=ATOL_APPROX_COMPARISON)
        probabilities = state.execute(follow_up, output_type="probs", mode=mode, shots=N_SHOTS)
        np.testing.assert_allclose(probabilities, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)

    def test_execute_shots_probs_batched_state(self):
        state = MixedGaussianState.from_polarizations(np.array([[1.0, 0.5], [-1.0, 1.0]]))
        np.random.seed(TEST_SEED)
        probabilities = state.execute([], output_type="probs", mode="parallel", shots=N_SHOTS)
        assert probabilities.shape == (2, 4)
        np.testing.assert_allclose(probabilities[0], [0.75, 0.25, 0.0, 0.0], atol=ATOL_APPROX_COMPARISON)
        np.testing.assert_allclose(probabilities[1], [0.0, 0.0, 1.0, 0.0], atol=ATOL_APPROX_COMPARISON)
        value = state.execute([], qml.PauliZ(1), "expval", mode="direct", shots=N_SHOTS)
        np.testing.assert_allclose(value, [0.5, 1.0], atol=ATOL_APPROX_COMPARISON)

    def test_execute_returns_none_without_output_type(self, reduced_case):
        state, _, follow_up, _ = reduced_case
        assert state.execute(follow_up) is None

    def test_execute_rejects_unknown_output_type_and_mode(self, reduced_case):
        state, _, follow_up, _ = reduced_case
        with pytest.raises(ValueError, match="Output type"):
            state.execute(follow_up, output_type="star_state")
        with pytest.raises(ValueError, match="execution mode"):
            state.execute(follow_up, output_type="probs", mode="fastest")

    def test_execute_rejects_state_preparations_and_foreign_wires(self, reduced_case):
        state, _, _, _ = reduced_case
        with pytest.raises(ValueError, match="state preparation"):
            state.execute([qml.BasisState(np.zeros(3, dtype=int), wires=range(3))], output_type="probs")
        with pytest.raises(ValueError, match="not wires of the state"):
            state.execute([CompRxRx(np.zeros(2), wires=[2, 3])], output_type="probs")

    def test_execute_device_kwargs_override(self, reduced_case):
        state, _, follow_up, evolved_density = reduced_case
        state.device_kwargs["r_dtype"] = torch.float64
        probabilities = state.execute(
            follow_up, output_type="probs", mode="direct", device_kwargs={"pfaffian_chunk_size": 2}
        )
        np.testing.assert_allclose(
            probabilities, np.diag(evolved_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_expval_and_probs_without_circuit(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        observable = qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliX(2)
        expected = np.trace(reduced_density @ qml.matrix(observable, wire_order=range(3))).real
        np.testing.assert_allclose(
            state.expval(observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            state.probs(), np.diag(reduced_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert state.execute([], observable, "expval") == pytest.approx(expected, abs=ATOL_SCALAR_COMPARISON)

    def test_expval_projector(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        projector = qml.Projector(np.array([1, 0]), wires=[0, 2])
        marginal = np.asarray(qml.math.partial_trace(reduced_density, [1]))
        expected = np.diag(marginal).real[2]
        np.testing.assert_allclose(
            state.expval(projector), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_expval_rejects_unsupported_observable(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        with pytest.raises(NotImplementedError, match="cannot be computed"):
            state.expval(qml.Hermitian(np.eye(2), wires=[0]) @ qml.Hermitian(np.eye(2), wires=[1]))

    def test_sample_requires_shots(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        with pytest.raises(ValueError, match="shots"):
            state.sample(None)

    def test_sample_batched_shape(self):
        state = MixedGaussianState.from_polarizations(np.array([[1.0, 0.5], [-1.0, 1.0]]))
        np.random.seed(TEST_SEED)
        samples = state.sample(11)
        assert samples.shape == (11, 2, 2)
        np.testing.assert_array_equal(samples[:, 0, 0], 0)
        np.testing.assert_array_equal(samples[:, 1, :], [[1, 0]] * 11)

    def test_evolve_matches_ensemble_and_dense(self, reduced_case):
        state, _, follow_up, evolved_density = reduced_case
        evolved = state.evolve(follow_up)
        assert evolved.wires == state.wires
        assert evolved.source_wires == state.source_wires
        np.testing.assert_allclose(
            evolved.covariance_matrix,
            self.dense_covariance(evolved_density, 3),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_evolve_with_torch_circuit_keeps_gradient(self):
        state = MixedGaussianState.from_polarizations(np.array([0.5, -0.2]))
        angles = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        evolved = state.evolve([CompRxRx(angles, wires=[0, 1])])
        value = evolved.expval(qml.PauliZ(0))
        assert isinstance(value, torch.Tensor) and value.requires_grad
        (gradient,) = torch.autograd.grad(value, angles)
        assert gradient.shape == (2,)

    def test_reduced_state_and_partial_trace_of_state(self, reduced_case):
        state, reduced_density, follow_up, evolved_density = reduced_case
        chained = state.evolve(follow_up).partial_trace([0])
        assert chained.wires.tolist() == [0, 1]
        assert chained.source_wires.tolist() == [2, 3]
        expected = np.asarray(qml.math.partial_trace(evolved_density, [0]))
        np.testing.assert_allclose(
            chained.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        kept = state.reduced_state([2, 1], reduced_wires=[5, 6])
        assert kept.wires.tolist() == [5, 6]
        assert kept.source_wires.tolist() == [2, 3]
        np.testing.assert_allclose(
            kept.covariance_matrix,
            state.covariance_matrix[2:, 2:],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_reduced_state_rejects_bad_wires(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        with pytest.raises(ValueError, match="not wires of this state"):
            state.reduced_state([3])
        with pytest.raises(ValueError, match="not wires of this state"):
            state.partial_trace([3])
        with pytest.raises(ValueError, match="At least one wire"):
            state.reduced_state([])

    def test_execute_parallel_batched_state(self):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        batched_angles = rng.normal(size=(3, 2))
        shared = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([CompRxRx(batched_angles, wires=[0, 1])] + shared)
        state = device.partial_trace([3])
        assert state.batch_shape == (3,)
        follow_up = self.brick_wall(range(3), 1, rng)
        observable = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliZ(2)
        values = {
            mode: np.asarray(state.execute(follow_up, observable, "expval", mode=mode))
            for mode in ("parallel", "sequential", "direct")
        }
        probabilities = np.asarray(state.execute(follow_up, output_type="probs", mode="parallel"))
        np.random.seed(TEST_SEED)
        samples = state.execute(follow_up, output_type="samples", mode="sequential", shots=5)
        assert samples.shape == (5, 3, 3)
        unitary = self.dense_unitary(follow_up, 3)
        for batch_index in range(3):
            operations = [CompRxRx(batched_angles[batch_index], wires=[0, 1])] + shared
            psi = self.dense_state(np.zeros(n_wires, dtype=int), operations, n_wires)
            reduced = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [3]))
            evolved = unitary @ reduced @ unitary.conj().T
            expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(3))).real
            for mode, value in values.items():
                np.testing.assert_allclose(
                    value[batch_index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
                )
            np.testing.assert_allclose(probabilities[batch_index], np.diag(evolved).real, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["sequential", "direct"])
    def test_execute_batched_follow_up_circuit(self, reduced_case, mode):
        state, reduced_density, _, _ = reduced_case
        batched_angles = np.random.default_rng(TEST_SEED).normal(size=(2, 2))
        follow_up = [CompRyRy(batched_angles, wires=[1, 2])]
        observable = qml.PauliZ(1) @ qml.PauliZ(2)
        values = np.asarray(state.execute(follow_up, observable, "expval", mode=mode))
        assert values.shape == (2,)
        for batch_index in range(2):
            unitary = self.dense_unitary([CompRyRy(batched_angles[batch_index], wires=[1, 2])], 3)
            evolved = unitary @ reduced_density @ unitary.conj().T
            expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(3))).real
            np.testing.assert_allclose(
                values[batch_index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_gradient_through_partial_trace_matches_dense(self, mode):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        angles = torch.tensor(rng.normal(size=(3, 2)), dtype=torch.float64, requires_grad=True)
        basis_state = np.array([1, 0, 0, 1])

        def build(parameters):
            return [
                CompRxRx(parameters[0], wires=[0, 1]),
                CompRyRy(parameters[1], wires=[1, 2]),
                CompRxRx(parameters[2], wires=[2, 3]),
                fSWAP(wires=[1, 2]),
            ]

        follow_up = self.brick_wall(range(2), 1, rng)
        observable = qml.PauliX(0) @ qml.PauliX(1) + 0.4 * qml.PauliZ(1)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([qml.BasisState(basis_state, wires=range(n_wires))] + build(angles))
        value = device.partial_trace([0, 3]).execute(follow_up, observable, "expval", mode=mode)
        (gradient,) = torch.autograd.grad(value, angles)

        qubit_device = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(qubit_device, interface="torch")
        def dense_circuit(parameters):
            qml.BasisState(basis_state, wires=range(n_wires))
            with qml.QueuingManager.stop_recording():
                operations = build(parameters)
            for operation in operations:
                qml.apply(operation)
            return qml.state()

        reference_angles = angles.detach().clone().requires_grad_()
        psi = dense_circuit(reference_angles)
        reduced = qml.math.partial_trace(torch.outer(psi, psi.conj()), [0, 3])
        unitary = torch.tensor(self.dense_unitary(follow_up, 2))
        evolved = unitary @ reduced @ unitary.conj().T
        reference = torch.trace(evolved @ torch.tensor(qml.matrix(observable, wire_order=range(2)))).real
        (reference_gradient,) = torch.autograd.grad(reference, reference_angles)
        torch.testing.assert_close(
            value.detach(), reference.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            gradient, reference_gradient, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
