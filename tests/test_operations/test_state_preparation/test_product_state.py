from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.utils.majorana import get_majorana


def _brute_force_lambda(psi: np.ndarray, n: int) -> np.ndarray:
    """Lambda_{mu nu} = i <psi| c_mu c_nu |psi> for a full state vector psi.

    The returned matrix is real and (2n, 2n) antisymmetric.
    """
    rho = np.outer(psi, np.conj(psi))
    cov = np.zeros((2 * n, 2 * n), dtype=complex)
    for mu in range(2 * n):
        for nu in range(2 * n):
            if mu == nu:
                continue
            cov[mu, nu] = 1j * np.trace(rho @ get_majorana(mu, n) @ get_majorana(nu, n))
    np.testing.assert_allclose(
        cov.imag,
        0.0,
        atol=1e-9,
        err_msg="brute-force Lambda has nonzero imaginary part",
    )
    return cov.real


def _per_qubit_to_full_vector(state: np.ndarray, n: int) -> np.ndarray:
    """Kronecker product of n per-qubit (alpha, beta) pairs into a 2^n vector.

    Accepts either flat (2n,) or matrix (n, 2) input.
    """
    pairs = state.reshape(n, 2)
    full = pairs[0]
    for k in range(1, n):
        full = np.kron(full, pairs[k])
    return full


def _random_product_state(n: int, seed: int = 0) -> np.ndarray:
    """A uniformly random pure product state as a flat (2n,) complex vector.

    Each qubit is drawn uniformly on the Bloch sphere via the
    (theta, phi) parametrisation.
    """
    rng = np.random.RandomState(seed)
    state = np.zeros(2 * n, dtype=complex)
    for k in range(n):
        theta = rng.uniform(0.0, np.pi)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        state[2 * k] = np.cos(theta / 2.0)
        state[2 * k + 1] = np.exp(1j * phi) * np.sin(theta / 2.0)
    return state


class TestProductState:
    def test_init_rejects_wrong_length_vector(self):
        with pytest.raises(ValueError, match="shape"):
            ProductState(np.zeros(3, dtype=complex), wires=[0, 1])

    def test_init_rejects_wrong_matrix_shape(self):
        with pytest.raises(ValueError, match="shape"):
            ProductState(np.zeros((2, 3), dtype=complex), wires=[0, 1])

    def test_init_rejects_non_unit_norm_qubit(self):
        # Second qubit has norm sqrt(2).
        bad = np.array([1, 0, 1, 1], dtype=complex)
        with pytest.raises(ValueError, match="unit norm"):
            ProductState(bad, wires=[0, 1])

    def test_init_norm_validation_can_be_disabled(self):
        """validate_norm=False must accept un-normalised inputs without raising."""
        bad = np.array([1, 0, 1, 1], dtype=complex)
        op = ProductState(bad, wires=[0, 1], validate_norm=False)
        assert op.data[0].shape == (2, 2)

    def test_flat_and_matrix_input_produce_same_internal_state(self):
        flat = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1, 1, 0], dtype=complex)
        matrix = flat.reshape(3, 2)

        op_flat = ProductState(flat, wires=[0, 1, 2])
        op_matrix = ProductState(matrix, wires=[0, 1, 2])

        assert op_flat.data[0].shape == (3, 2)
        assert op_matrix.data[0].shape == (3, 2)
        np.testing.assert_allclose(
            np.asarray(op_flat.data[0]),
            np.asarray(op_matrix.data[0]),
        )

        # ndim_params should reflect the canonical 2-D layout.
        assert op_flat.ndim_params == (2,)

    def test_flat_and_matrix_input_produce_same_covariance(self):
        flat = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1, 1, 0], dtype=complex)
        matrix = flat.reshape(3, 2)
        cov_flat = ProductState(flat, wires=[0, 1, 2]).covariance_matrix
        cov_matrix = ProductState(matrix, wires=[0, 1, 2]).covariance_matrix
        torch.testing.assert_close(cov_flat, cov_matrix)

    def test_state_vector_matches_explicit_kron(self):
        n = 3
        # |+> on q0, |1> on q1, (|0>+i|1>)/sqrt(2) on q2
        state = np.array(
            [
                1 / np.sqrt(2),
                1 / np.sqrt(2),
                0,
                1,
                1 / np.sqrt(2),
                1j / np.sqrt(2),
            ],
            dtype=complex,
        )
        op = ProductState(state, wires=[0, 1, 2])
        psi_op = np.asarray(op.state_vector()).reshape(-1)
        psi_ref = _per_qubit_to_full_vector(state, n)
        np.testing.assert_allclose(psi_op, psi_ref, atol=1e-12)

    def test_state_vector_permutes_under_wire_order(self):
        # |0>|1> on wires [0, 1]; asking for wire_order [1, 0] should swap.
        op = ProductState(np.array([1, 0, 0, 1], dtype=complex), wires=[0, 1])
        psi_ab = np.asarray(op.state_vector(wire_order=[0, 1])).reshape(-1)  # |01>
        psi_ba = np.asarray(op.state_vector(wire_order=[1, 0])).reshape(-1)  # |10>
        np.testing.assert_allclose(psi_ab, np.array([0, 1, 0, 0], dtype=complex))
        np.testing.assert_allclose(psi_ba, np.array([0, 0, 1, 0], dtype=complex))

    def test_state_vector_embeds_into_larger_wire_order(self):
        # |1> on wire 1; ask for wire_order [0, 1, 2]: extra wires padded with |0>
        op = ProductState(np.array([0, 1], dtype=complex), wires=[1])
        psi = np.asarray(op.state_vector(wire_order=[0, 1, 2])).reshape(-1)
        # Expected: |0>|1>|0> = index 010_b = 2
        expected = np.zeros(8, dtype=complex)
        expected[0b010] = 1.0
        np.testing.assert_allclose(psi, expected, atol=1e-12)

    def test_state_vector_rejects_incomplete_wire_order(self):
        op = ProductState(np.array([1, 0, 0, 1], dtype=complex), wires=[0, 1])
        with pytest.raises(ValueError, match="must contain"):
            op.state_vector(wire_order=[0])

    @pytest.mark.parametrize(
        "bits",
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1, 0],
            [1, 0, 1, 1],
        ],
    )
    def test_is_basis_state_true_on_basis_states(self, bits):
        n = len(bits)
        state = np.zeros((n, 2), dtype=complex)
        for k, b in enumerate(bits):
            state[k, b] = 1.0
        op = ProductState(state, wires=range(n))
        assert op.is_basis_state is True
        np.testing.assert_array_equal(
            np.asarray(op.as_basis_state().parameters[0]),
            np.asarray(bits),
        )

    def test_is_basis_state_true_under_global_phases(self):
        # |0>x-|1>: still a basis state up to global phase per qubit.
        op = ProductState(np.array([1, 0, 0, -1], dtype=complex), wires=[0, 1])
        assert op.is_basis_state is True
        np.testing.assert_array_equal(np.asarray(op.as_basis_state().parameters[0]), np.array([0, 1]))

    def test_is_basis_state_false_on_superposition(self):
        state = np.array([1, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        op = ProductState(state, wires=[0, 1])
        assert op.is_basis_state is False

    def test_as_basis_state_raises_on_superposition(self):
        state = np.array([1, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        op = ProductState(state, wires=[0, 1])
        with pytest.raises(ValueError, match="not a computational-basis state"):
            op.as_basis_state()

    def test_covariance_matrix_shape_and_dtype(self):
        op = ProductState(_random_product_state(3, seed=0), wires=range(3))
        cov = op.covariance_matrix
        assert isinstance(cov, torch.Tensor)
        assert cov.shape == (6, 6)
        assert cov.dtype in (torch.float32, torch.float64)

    def test_covariance_matrix_is_real_antisymmetric(self):
        op = ProductState(_random_product_state(4, seed=1), wires=range(4))
        cov = op.covariance_matrix
        torch.testing.assert_close(cov, -cov.T)

    def test_covariance_matrix_is_cached(self):
        op = ProductState(_random_product_state(3, seed=0), wires=[0, 1, 2])
        cov1 = op.covariance_matrix
        cov2 = op.covariance_matrix
        assert cov1 is cov2, "covariance_matrix should be cached"

    @pytest.mark.parametrize(
        "bits",
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1, 0],
            [1, 0, 1, 1],
        ],
    )
    def test_covariance_matrix_basis_states_match_brute_force(self, bits):
        n = len(bits)
        state = np.zeros((n, 2), dtype=complex)
        for k, b in enumerate(bits):
            state[k, b] = 1.0

        op = ProductState(state, wires=range(n))
        cov_op = op.covariance_matrix.detach().cpu().numpy()
        cov_ref = _brute_force_lambda(_per_qubit_to_full_vector(state.reshape(2 * n), n), n)
        np.testing.assert_allclose(cov_op, cov_ref, atol=1e-9)

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_covariance_matrix_random_product_states_match_brute_force(self, n, seed):
        state = _random_product_state(n, seed=seed)
        op = ProductState(state, wires=range(n))
        cov_op = op.covariance_matrix.detach().cpu().numpy()
        cov_ref = _brute_force_lambda(_per_qubit_to_full_vector(state, n), n)
        np.testing.assert_allclose(cov_op, cov_ref, atol=1e-9)

    @pytest.mark.parametrize(
        "bits",
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 0, 1, 1],
        ],
    )
    def test_covariance_matrix_basis_state_is_orthogonal(self, bits):
        """Lambda^T Lambda = I for fermionic Gaussian states; computational
        basis states are Gaussian. (General product states are not, so we
        only check this property here.)"""
        n = len(bits)
        state = np.zeros((n, 2), dtype=complex)
        for k, b in enumerate(bits):
            state[k, b] = 1.0
        op = ProductState(state, wires=range(n))
        cov = op.covariance_matrix
        eye = torch.eye(2 * n, dtype=cov.dtype, device=cov.device)
        torch.testing.assert_close(cov.T @ cov, eye, atol=1e-9, rtol=0.0)

    def test_init_with_real_state_casts_to_complex(self):
        state = np.array([1.0, 0.0, 0.0, 1.0])
        op = ProductState(state, wires=[0, 1])
        assert np.iscomplexobj(np.asarray(op.data[0]))

    def test_compute_decomposition_returns_state_prep(self):
        state = np.array(
            [[1, 0], [0, 1]],
            dtype=complex,
        )
        decomp = ProductState.compute_decomposition(state, wires=[0, 1])
        assert len(decomp) == 1
        assert isinstance(decomp[0], qml.StatePrep)

    def test_from_basis_state_from_int_array(self):
        op = ProductState.from_basis_state([0, 1], wires=[0, 1])
        assert op.is_basis_state
        np.testing.assert_array_equal(np.asarray(op.as_basis_state().parameters[0]), [0, 1])

    def test_from_basis_state_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            ProductState.from_basis_state([0, 1, 0], wires=[0, 1])

    def test_as_torch_returns_same_tensor_if_already_tensor(self):
        state = np.array([[1, 0], [0, 1]], dtype=complex)
        op = ProductState(state, wires=[0, 1])
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = op._as_torch(t)
        assert result is t

    def test_inter_qubit_block_vanishes_when_z_eigenstate_in_between(self):
        # |0> |+> |0>: qubit 1 has z_1 = 0, so the parity string between
        # qubits 0 and 2 is identically zero -> their inter-qubit 2x2 block
        # in Lambda_0 must vanish.
        state = np.array(
            [
                1,
                0,  # |0>
                1 / np.sqrt(2),
                1 / np.sqrt(2),  # |+>
                1,
                0,  # |0>
            ],
            dtype=complex,
        )
        op = ProductState(state, wires=[0, 1, 2])
        cov = op.covariance_matrix
        # Inter-qubit block between qubits 0 and 2 = Lambda[0:2, 4:6]
        block_02 = cov[0:2, 4:6]
        torch.testing.assert_close(
            block_02,
            torch.zeros_like(block_02),
            atol=1e-9,
            rtol=0.0,
        )


def _random_batched_product_state(batch: int, n: int, seed: int = 0) -> np.ndarray:
    """A stack of ``batch`` random product states as a (batch, 2n) array."""
    return np.stack([_random_product_state(n, seed=seed + b) for b in range(batch)])


class TestProductStateBatched:
    def test_batch_size_from_flat_input(self):
        state = _random_batched_product_state(batch=4, n=3, seed=0)
        op = ProductState(state, wires=range(3))
        assert op.data[0].shape == (4, 3, 2)
        assert op.batch_size == 4

    def test_batch_size_from_matrix_input(self):
        state = _random_batched_product_state(batch=4, n=3, seed=0).reshape(4, 3, 2)
        op = ProductState(state, wires=range(3))
        assert op.data[0].shape == (4, 3, 2)
        assert op.batch_size == 4

    def test_unbatched_input_has_no_batch_size(self):
        op = ProductState(_random_product_state(3, seed=0), wires=range(3))
        assert op.batch_size is None
        assert op.data[0].shape == (3, 2)

    def test_batched_flat_and_matrix_produce_same_internal_state(self):
        flat = _random_batched_product_state(batch=3, n=2, seed=7)
        matrix = flat.reshape(3, 2, 2)
        op_flat = ProductState(flat, wires=[0, 1])
        op_matrix = ProductState(matrix, wires=[0, 1])
        np.testing.assert_allclose(
            np.asarray(op_flat.data[0]),
            np.asarray(op_matrix.data[0]),
        )

    def test_init_rejects_wrong_batched_shape(self):
        # (batch, 5) does not match 2*n_wires = 4 for two wires.
        with pytest.raises(ValueError, match="shape"):
            ProductState(np.zeros((3, 5), dtype=complex), wires=[0, 1])

    def test_init_rejects_non_unit_norm_in_batch(self):
        state = _random_batched_product_state(batch=2, n=2, seed=0)
        state[1, 2:] = [1.0, 1.0]  # break normalisation of qubit 1 in sample 1
        with pytest.raises(ValueError, match="unit norm"):
            ProductState(state, wires=[0, 1])

    def test_batched_covariance_shape(self):
        state = _random_batched_product_state(batch=5, n=3, seed=1)
        cov = ProductState(state, wires=range(3)).covariance_matrix
        assert cov.shape == (5, 6, 6)

    def test_batched_covariance_matches_per_state(self):
        batch, n = 4, 3
        state = _random_batched_product_state(batch=batch, n=n, seed=2)
        cov_batched = ProductState(state, wires=range(n)).covariance_matrix
        for b in range(batch):
            cov_single = ProductState(state[b], wires=range(n)).covariance_matrix
            torch.testing.assert_close(cov_batched[b], cov_single)

    def test_batched_covariance_matches_brute_force(self):
        batch, n = 3, 3
        state = _random_batched_product_state(batch=batch, n=n, seed=5)
        cov_batched = ProductState(state, wires=range(n)).covariance_matrix.detach().cpu().numpy()
        for b in range(batch):
            cov_ref = _brute_force_lambda(_per_qubit_to_full_vector(state[b], n), n)
            np.testing.assert_allclose(cov_batched[b], cov_ref, atol=1e-9)

    def test_batched_covariance_is_antisymmetric(self):
        state = _random_batched_product_state(batch=4, n=4, seed=3)
        cov = ProductState(state, wires=range(4)).covariance_matrix
        torch.testing.assert_close(cov, -qml.math.einsum("...ij->...ji", cov))

    def test_batched_state_vector_matches_per_state(self):
        batch, n = 4, 3
        state = _random_batched_product_state(batch=batch, n=n, seed=4)
        sv_batched = np.asarray(ProductState(state, wires=range(n)).state_vector())
        assert sv_batched.shape == (batch,) + (2,) * n
        for b in range(batch):
            ref = _per_qubit_to_full_vector(state[b], n).reshape((2,) * n)
            np.testing.assert_allclose(sv_batched[b], ref, atol=1e-12)

    def test_batched_state_vector_permutes_under_wire_order(self):
        # Two samples: |01> and |10> on wires [0, 1].
        state = np.array(
            [
                [1, 0, 0, 1],  # |0>|1>
                [0, 1, 1, 0],  # |1>|0>
            ],
            dtype=complex,
        )
        op = ProductState(state, wires=[0, 1])
        psi_ba = np.asarray(op.state_vector(wire_order=[1, 0])).reshape(2, -1)
        # Swapping wires turns |01> -> |10> and |10> -> |01>.
        np.testing.assert_allclose(psi_ba[0], np.array([0, 0, 1, 0], dtype=complex))
        np.testing.assert_allclose(psi_ba[1], np.array([0, 1, 0, 0], dtype=complex))

    def test_batched_state_vector_embeds_into_larger_wire_order(self):
        # Two single-qubit samples on wire 1: |0> and |1>.
        state = np.array([[1, 0], [0, 1]], dtype=complex)
        op = ProductState(state, wires=[1])
        psi = np.asarray(op.state_vector(wire_order=[0, 1, 2])).reshape(2, -1)
        expected0 = np.zeros(8, dtype=complex)
        expected0[0b000] = 1.0  # |0>|0>|0>
        expected1 = np.zeros(8, dtype=complex)
        expected1[0b010] = 1.0  # |0>|1>|0>
        np.testing.assert_allclose(psi[0], expected0, atol=1e-12)
        np.testing.assert_allclose(psi[1], expected1, atol=1e-12)

    def test_batched_is_basis_state(self):
        # First sample is a basis state |01>, second is a superposition.
        state = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            ],
            dtype=complex,
        )
        op = ProductState(state, wires=[0, 1])
        result = np.asarray(op.is_basis_state)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [True, False])

    def test_batched_basis_state_bits(self):
        state = np.array(
            [
                [1, 0, 0, 1],  # |01>
                [0, 1, 1, 0],  # |10>
            ],
            dtype=complex,
        )
        op = ProductState(state, wires=[0, 1])
        bits = op.basis_state_bits()
        assert bits.shape == (2, 2)
        np.testing.assert_array_equal(bits, [[0, 1], [1, 0]])

    def test_batched_as_basis_state_raises(self):
        # qml.BasisState has no batch dimension; a batched ProductState can't
        # be collapsed to a single BasisState.
        state = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=complex)
        op = ProductState(state, wires=[0, 1])
        with pytest.raises(ValueError, match="batched"):
            op.as_basis_state()

    def test_batched_basis_state_bits_raises_when_any_superposition(self):
        state = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            ],
            dtype=complex,
        )
        op = ProductState(state, wires=[0, 1])
        with pytest.raises(ValueError, match="not a computational-basis state"):
            op.basis_state_bits()

    def test_from_basis_state_batched(self):
        bits = [[0, 1], [1, 0], [1, 1]]
        op = ProductState.from_basis_state(bits, wires=[0, 1])
        assert op.batch_size == 3
        np.testing.assert_array_equal(op.basis_state_bits(), bits)

    def test_compute_decomposition_batched(self):
        state = _random_batched_product_state(batch=3, n=2, seed=0)
        decomp = ProductState.compute_decomposition(state, wires=[0, 1])
        assert len(decomp) == 1
        assert isinstance(decomp[0], qml.StatePrep)
        assert decomp[0].batch_size == 3
