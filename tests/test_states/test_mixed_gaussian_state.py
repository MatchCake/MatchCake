import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import BatchHamiltonian, MixedGaussianState, NonInteractingFermionicDevice
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
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
        return state, reduced_density

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
        assert repr(state) == "MixedGaussianState(num_wires=2, batch_shape=())"

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
        reconstructed = state.sptm.T @ block_diagonal_covariance(state.mode_polarizations) @ state.sptm
        np.testing.assert_allclose(reconstructed, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        assert state.atol == 1e-6

    def test_atol_controls_purity_detection(self):
        polarizations = [1.0 - 1e-5, -1.0]
        assert not MixedGaussianState.from_polarizations(polarizations, atol=1e-8).is_pure
        assert MixedGaussianState.from_polarizations(polarizations, atol=1e-4).is_pure

    def test_entropy_matches_binary_entropy_and_base(self):
        state = MixedGaussianState.from_polarizations([0.0, 1.0, 0.5])
        expected = np.log(2) + (-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
        np.testing.assert_allclose(state.entropy(), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(
            state.entropy(base=2), expected / np.log(2), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_pure_basis_state(self):
        state = MixedGaussianState.from_polarizations([1.0, -1.0, 1.0])
        assert state.is_pure
        # The normal modes may be permuted by the decomposition, but the state is the basis state |010>.
        expected_probs = np.zeros(8)
        expected_probs[0b010] = 1.0
        np.testing.assert_allclose(state.probs(), expected_probs, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(state.purity, 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(state.entropy(), 0.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_block_diagonal_state_probs_and_purity(self):
        polarizations = np.array([0.6, -1.0, -0.2])
        state = MixedGaussianState.from_polarizations(polarizations)
        assert not state.is_pure
        # The covariance matrix is already block diagonal, so the state is diagonal in the computational basis.
        bits = np.array([[(index >> shift) & 1 for shift in (2, 1, 0)] for index in range(8)])
        expected_probs = np.prod(0.5 * (1.0 + (1 - 2 * bits) * polarizations), axis=-1)
        np.testing.assert_allclose(
            state.probs(), expected_probs, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            state.purity, (1 + 0.36) / 2 * (1 + 0.04) / 2, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_batched_polarizations(self):
        polarizations = np.array([[1.0, 0.5], [-0.3, 1.0]])
        state = MixedGaussianState.from_polarizations(polarizations)
        assert state.batch_shape == (2,)
        np.testing.assert_array_equal(state.is_pure, [False, False])
        assert state.entropy().shape == (2,)
        assert state.purity.shape == (2,)

    def test_density_matrix_matches_qubit_partial_trace(self, reduced_case):
        state, reduced_density = reduced_case
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
        state, reduced_density = reduced_case
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
        state, reduced_density = reduced_case
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

    def test_expval_and_probs_without_circuit(self, reduced_case):
        state, reduced_density = reduced_case
        observable = qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliX(2)
        expected = np.trace(reduced_density @ qml.matrix(observable, wire_order=range(3))).real
        np.testing.assert_allclose(
            state.expval(observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            state.probs(), np.diag(reduced_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_expval_of_sum_matches_dense(self, reduced_case):
        state, reduced_density = reduced_case
        observable = qml.PauliX(0) @ qml.PauliY(1) + 0.5 * qml.PauliZ(2) + 0.3 * qml.PauliX(1) @ qml.PauliX(2)
        expected = np.trace(reduced_density @ qml.matrix(observable, wire_order=range(3))).real
        np.testing.assert_allclose(
            state.expval(observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_expval_batch_hamiltonian_matches_dense(self, reduced_case):
        state, reduced_density = reduced_case
        terms = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(1) @ qml.PauliX(2)]
        coefficients = [0.2, -0.7]
        expected = [
            coefficient * np.trace(reduced_density @ qml.matrix(term, wire_order=range(3))).real
            for coefficient, term in zip(coefficients, terms)
        ]
        np.testing.assert_allclose(
            state.expval(BatchHamiltonian(coefficients, terms)),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_marginal_probs_match_dense(self, reduced_case):
        state, reduced_density = reduced_case
        marginal = np.asarray(qml.math.partial_trace(reduced_density, [1]))
        np.testing.assert_allclose(
            state.probs(wires=[0, 2]), np.diag(marginal).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_expval_projector(self, reduced_case):
        state, reduced_density = reduced_case
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

    def test_samples_follow_probabilities(self, reduced_case):
        state, reduced_density = reduced_case
        np.random.seed(TEST_SEED)
        samples = state.sample(N_SHOTS)
        assert samples.shape == (N_SHOTS, 3)
        indices = samples @ (2 ** np.arange(3)[::-1])
        frequencies = np.bincount(indices, minlength=8) / N_SHOTS
        np.testing.assert_allclose(frequencies, np.diag(reduced_density).real, atol=ATOL_APPROX_COMPARISON)

    def test_reduced_state_and_partial_trace_of_state(self, reduced_case):
        state, reduced_density = reduced_case
        chained = state.partial_trace([0])
        assert chained.wires.tolist() == [0, 1]
        assert chained.source_wires.tolist() == [2, 3]
        expected = np.asarray(qml.math.partial_trace(reduced_density, [0]))
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

    def test_gradient_through_partial_trace_matches_dense(self):
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

        observable = qml.PauliX(0) @ qml.PauliX(1) + 0.4 * qml.PauliZ(1)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([qml.BasisState(basis_state, wires=range(n_wires))] + build(angles))
        value = device.partial_trace([0, 3]).expval(observable)
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
        reference = torch.trace(reduced @ torch.tensor(qml.matrix(observable, wire_order=range(2)))).real
        (reference_gradient,) = torch.autograd.grad(reference, reference_angles)
        torch.testing.assert_close(
            value.detach(), reference.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            gradient, reference_gradient, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
