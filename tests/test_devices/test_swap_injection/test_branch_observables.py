import itertools

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.devices.nif_device import NonInteractingFermionicDevice
from matchcake.devices.probability_strategies.product_state_strategy import (
    ProductStateProbabilityStrategy,
)
from matchcake.devices.swap_injection import (
    SwapBranchState,
    basis_state_probability,
    basis_states_probabilities,
    hamiltonian_expval,
    transition_cov,
)
from matchcake.devices.swap_injection.branch_observables import _pair_pfaffian_grid
from matchcake.utils._pfaffian_family import OutcomeFamily

from ...configs import ATOL_MATRIX_COMPARISON, ATOL_SCALAR_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED
from . import _oracle as oracle


class TestBranchObservables:
    """Branch observables -- ports ``verify_branch_formulas.py``."""

    @staticmethod
    def _gaussian_states(n, cs, rng, count=3):
        """A pool of Gaussian pure states, including one occupation-projected (still Gaussian) state."""
        states = []
        for _ in range(count):
            unitary = oracle.random_gaussian_unitary(n, cs, rng)
            bits = rng.integers(0, 2, size=n)
            states.append(unitary @ oracle.basis_state(n, bits))
        num = oracle.number_ops(n, cs)
        projected = num[0] @ num[1] @ states[0]
        states.append(projected / np.linalg.norm(projected))
        return states

    @staticmethod
    def _one_swap(rng, n=3):
        cs = oracle.majoranas(n)
        x0 = [1, 1, 0]
        gate1 = oracle.random_gaussian_unitary(n, cs, rng)
        gate2 = oracle.random_gaussian_unitary(n, cs, rng)
        psi = gate2 @ oracle.swap_matrix(n, 0, 1) @ gate1 @ oracle.basis_state(n, x0)
        lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)
        state = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        state.apply_matchgate_sptm(oracle.sptm_of(gate1, cs, n))
        state.apply_swap(0, 1)
        state.apply_matchgate_sptm(oracle.sptm_of(gate2, cs, n))
        return state, psi, n

    def test_transition_cov_diagonal_recovers_covariance(self):
        rng = np.random.default_rng(0)
        n = 3
        cs = oracle.majoranas(n)
        phi = oracle.random_gaussian_unitary(n, cs, rng) @ oracle.basis_state(n, [1, 0, 1])
        covariance, _ = oracle.phys_cov_disp(phi, cs)
        gamma = np.asarray(transition_cov(covariance, covariance))
        np.testing.assert_allclose(gamma, covariance, atol=ATOL_MATRIX_COMPARISON)

    def test_transition_cov_is_antisymmetric_with_zero_diagonal(self):
        rng = np.random.default_rng(1)
        n = 3
        cs = oracle.majoranas(n)
        a = oracle.random_gaussian_unitary(n, cs, rng) @ oracle.basis_state(n, [1, 0, 1])
        b = oracle.random_gaussian_unitary(n, cs, rng) @ oracle.basis_state(n, [0, 1, 1])
        cov_a, _ = oracle.phys_cov_disp(a, cs)
        cov_b, _ = oracle.phys_cov_disp(b, cs)
        gamma = np.asarray(transition_cov(cov_a, cov_b))
        np.testing.assert_allclose(gamma + gamma.T, 0.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(np.diagonal(gamma), 0.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("m", [2, 4, 6])
    def test_transition_wick(self, m):
        rng = np.random.default_rng(7)
        n = 3
        cs = oracle.majoranas(n)
        d = 2 * n
        states = self._gaussian_states(n, cs, rng)
        for a, b in itertools.combinations(range(len(states)), 2):
            psi_a, psi_b = states[a], states[b]
            overlap = psi_a.conj() @ psi_b
            if abs(overlap) < 1e-9:
                continue
            cov_a, _ = oracle.phys_cov_disp(psi_a, cs)
            cov_b, _ = oracle.phys_cov_disp(psi_b, cs)
            gamma = np.asarray(transition_cov(cov_a, cov_b))
            for _ in range(4):
                support = np.sort(rng.choice(d, size=m, replace=False))
                operator = np.eye(2**n, dtype=complex)
                for mu in support:
                    operator = operator @ cs[mu]
                lhs = (psi_a.conj() @ operator @ psi_b) / overlap
                rhs = (1j) ** (-(m // 2)) * oracle.pfaffian(gamma[np.ix_(support, support)])
                np.testing.assert_allclose(lhs, rhs, atol=ATOL_MATRIX_COMPARISON)

    def test_projector_matrix_element(self):
        rng = np.random.default_rng(7)
        n = 3
        cs = oracle.majoranas(n)
        num = oracle.number_ops(n, cs)
        states = self._gaussian_states(n, cs, rng)
        for a, b in [(0, 1), (0, 3), (1, 2)]:
            psi_a, psi_b = states[a], states[b]
            overlap = psi_a.conj() @ psi_b
            if abs(overlap) < 1e-9:
                continue
            cov_a, _ = oracle.phys_cov_disp(psi_a, cs)
            cov_b, _ = oracle.phys_cov_disp(psi_b, cs)
            gamma = np.asarray(transition_cov(cov_a, cov_b))
            for y in itertools.product((0, 1), repeat=n):
                projector = np.eye(2**n, dtype=complex)
                for k in range(n):
                    projector = projector @ (num[k] if y[k] else (np.eye(2**n) - num[k]))
                lhs = (psi_a.conj() @ projector @ psi_b) / overlap
                lambda_y = ProductStateProbabilityStrategy.build_lambda_y(np.array(y), n)
                rhs = 2.0**-n * np.prod(2 * np.array(y) - 1) * oracle.pfaffian(gamma + lambda_y)
                np.testing.assert_allclose(lhs, rhs, atol=ATOL_MATRIX_COMPARISON)

    def test_one_swap_probabilities_sum_to_one_and_match(self):
        rng = np.random.default_rng(7)
        state, psi, n = self._one_swap(rng)
        total = 0.0
        for y in itertools.product((0, 1), repeat=n):
            got = float(basis_state_probability(state.cov, state.weights, np.array(y)))
            total += got
            np.testing.assert_allclose(
                got, abs(oracle.basis_state(n, y).conj() @ psi) ** 2, atol=ATOL_MATRIX_COMPARISON
            )
        np.testing.assert_allclose(total, 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_one_swap_hamiltonian_matches_statevector(self):
        rng = np.random.default_rng(7)
        state, psi, n = self._one_swap(rng)
        terms = [(rng.normal(), "".join(rng.choice(list("IXYZ"), size=n))) for _ in range(5)]
        hamiltonian = qml.Hamiltonian([c for c, _ in terms], [qml.pauli.string_to_pauli_word(p) for _, p in terms])
        dense = sum(c * oracle.kron_list([oracle.PAULI_MAP[ch] for ch in p]) for c, p in terms)
        exact = (psi.conj() @ dense @ psi).real
        got = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=None))
        np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_hamiltonian_with_identity_term(self):
        # The rank-0 (identity) Pauli term takes the empty-support branch in hamiltonian_expval.
        rng = np.random.default_rng(7)
        state, psi, n = self._one_swap(rng)
        hamiltonian = qml.Hamiltonian([0.7, 0.5], [qml.Identity(0), qml.PauliZ(0) @ qml.PauliZ(1)])
        dense = 0.7 * np.eye(2**n) + 0.5 * oracle.kron_list([oracle.Z, oracle.Z, oracle.I2])
        exact = (psi.conj() @ dense @ psi).real
        got = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=None))
        np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_term_dense_hamiltonian_matches_statevector(self):
        # A term-dense random Hamiltonian (many words, mixed support sizes, identity included) exercises the
        # multi-group batched Pfaffian path against the exact statevector expectation.
        rng = np.random.default_rng(11)
        state, psi, n = self._one_swap(rng)
        words = ["".join(rng.choice(list("IXYZ"), size=n)) for _ in range(40)] + ["I" * n]
        terms = [(rng.normal(), word) for word in words]
        hamiltonian = qml.Hamiltonian([c for c, _ in terms], [qml.pauli.string_to_pauli_word(p) for _, p in terms])
        dense = sum(c * oracle.kron_list([oracle.PAULI_MAP[ch] for ch in p]) for c, p in terms)
        exact = (psi.conj() @ dense @ psi).real
        got = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=None))
        np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_hamiltonian_expval_pfaffian_chunk_size_invariance(self):
        # pfaffian_chunk_size bounds the batched Pfaffian reduction (PR #147); the result must not change.
        rng = np.random.default_rng(13)
        state, psi, n = self._one_swap(rng)
        terms = [(rng.normal(), "".join(rng.choice(list("IXYZ"), size=n))) for _ in range(12)]
        hamiltonian = qml.Hamiltonian([c for c, _ in terms], [qml.pauli.string_to_pauli_word(p) for _, p in terms])
        unchunked = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=None))
        chunked = float(
            hamiltonian_expval(
                state.cov, state.weights, hamiltonian, list(range(n)), marker=None, pfaffian_chunk_size=1
            )
        )
        np.testing.assert_allclose(chunked, unchunked, atol=ATOL_SCALAR_COMPARISON)

    @staticmethod
    def _multi_branch_state(seed, n=3, n_swaps=2):
        """A non-degenerate multi-branch state (chi >= 2) and its statevector, ported to the oracle."""
        for offset in range(80):
            rng = np.random.default_rng(seed + 1000 * offset)
            cs = oracle.majoranas(n)
            x0 = rng.integers(0, 2, size=n).tolist()
            lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)
            state = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
            psi = oracle.basis_state(n, x0)
            for _ in range(n_swaps):
                gate = oracle.random_gaussian_unitary(n, cs, rng)
                state.apply_matchgate_sptm(oracle.sptm_of(gate, cs, n))
                psi = gate @ psi
                j = int(rng.integers(0, n - 1))
                state.apply_swap(j, j + 1)
                psi = oracle.swap_matrix(n, j, j + 1) @ psi
            gate = oracle.random_gaussian_unitary(n, cs, rng)
            state.apply_matchgate_sptm(oracle.sptm_of(gate, cs, n))
            psi = gate @ psi
            if not state.degenerate and state.cov.shape[0] >= 2:
                return state, psi, n
        raise RuntimeError("no non-degenerate multi-branch state found")

    @staticmethod
    def _deterministic_wire_state(seed, n=3):
        """A two-branch state whose last wire stays deterministic, so the complex grid hits exact-zero pivots."""
        rng = np.random.default_rng(seed)
        cs = oracle.majoranas(n)
        cs2 = oracle.majoranas(2)
        x0 = [1, 1] + [0] * (n - 2)
        lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)
        state = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        embedded = np.kron(oracle.random_gaussian_unitary(2, cs2, rng), np.eye(2 ** (n - 2)))  # acts on wires 0, 1
        state.apply_matchgate_sptm(oracle.sptm_of(embedded, cs, n))
        psi = embedded @ oracle.basis_state(n, x0)
        state.apply_swap(0, 1)
        psi = oracle.swap_matrix(n, 0, 1) @ psi
        return state, psi, n

    @staticmethod
    def _statevector_probabilities(psi, n):
        binary = NonInteractingFermionicDevice.states_to_binary(np.arange(2**n), n)
        return np.array([abs(oracle.basis_state(n, binary[i]).conj() @ psi) ** 2 for i in range(2**n)])

    @staticmethod
    def _per_outcome_probabilities(cov, weights, measured_qubits):
        k = len(measured_qubits)
        binary = NonInteractingFermionicDevice.states_to_binary(np.arange(2**k), k)
        return np.stack(
            [np.asarray(basis_state_probability(cov, weights, binary[i], measured_qubits)) for i in range(2**k)]
        )

    @pytest.mark.parametrize("hermitian_halving", [True, False])
    @pytest.mark.parametrize("seed, n, n_swaps", [(0, 3, 2), (1, 3, 3), (2, 4, 2), (3, 4, 3)])
    def test_full_distribution_matches_per_outcome_loop(self, seed, n, n_swaps, hermitian_halving):
        # The one-shot tree (change 6 hoist + change 7 halving) reproduces the per-outcome loop exactly.
        state, _, n = self._multi_branch_state(seed, n=n, n_swaps=n_swaps)
        measured = list(range(n))
        loop = self._per_outcome_probabilities(state.cov, state.weights, measured)
        tree = np.asarray(
            basis_states_probabilities(state.cov, state.weights, measured, hermitian_halving=hermitian_halving)
        )
        np.testing.assert_allclose(tree, loop, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_full_distribution_matches_statevector(self, seed):
        state, psi, n = self._multi_branch_state(seed, n=3, n_swaps=2)
        tree = np.asarray(basis_states_probabilities(state.cov, state.weights, list(range(n))))
        np.testing.assert_allclose(tree, self._statevector_probabilities(psi, n), atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("measured", [[0, 1], [1, 2], [0, 2], [0]])
    def test_full_distribution_marginal_matches_loop(self, measured):
        # A marginal over a subset of wires uses the same tree path, on the measured Majorana block.
        state, _, n = self._multi_branch_state(1, n=3, n_swaps=2)
        loop = self._per_outcome_probabilities(state.cov, state.weights, measured)
        tree = np.asarray(basis_states_probabilities(state.cov, state.weights, measured))
        np.testing.assert_allclose(tree, loop, atol=ATOL_MATRIX_COMPARISON)

    def test_hermitian_pair_grid_symmetry(self):
        # Pf_{ba}(y) = conj(Pf_{ab}(y)) and the diagonal is real, so the reconstructed grid is Hermitian per outcome.
        state, _, n = self._multi_branch_state(2, n=3, n_swaps=2)
        mode_index = torch.arange(2 * n)
        grid = _pair_pfaffian_grid(
            torch.as_tensor(np.asarray(state.cov)), mode_index, torch.complex128, hermitian_halving=True
        )  # (chi, chi, 2^k)
        torch.testing.assert_close(
            grid, grid.transpose(0, 1).conj(), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        chi = grid.shape[0]
        diagonal = grid[torch.arange(chi), torch.arange(chi)]
        torch.testing.assert_close(
            diagonal.imag, torch.zeros_like(diagonal.imag), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("hermitian_halving", [True, False])
    def test_full_distribution_is_a_distribution(self, hermitian_halving):
        # Probabilities are real, nonnegative and sum to one.
        state, _, n = self._multi_branch_state(0, n=3, n_swaps=2)
        probs = np.asarray(
            basis_states_probabilities(state.cov, state.weights, list(range(n)), hermitian_halving=hermitian_halving)
        )
        assert probs.dtype.kind == "f"
        assert np.all(probs >= -ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_halving_matches_full_grid(self):
        # Change 7 (halving) and the plain full grid give identical probabilities.
        state, _, n = self._multi_branch_state(3, n=4, n_swaps=3)
        measured = list(range(n))
        halved = np.asarray(basis_states_probabilities(state.cov, state.weights, measured, hermitian_halving=True))
        full = np.asarray(basis_states_probabilities(state.cov, state.weights, measured, hermitian_halving=False))
        np.testing.assert_allclose(halved, full, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_exact_zero_pivot_input_matches_reference(self, seed):
        # A deterministic-wire input makes the complex grid hit exact-zero intermediate pivots (this test
        # confirms that code path is reached and the tree stays exact vs the pivoted loop and statevector).
        # For physical branch pairs the spurious-zero lanes are genuine zero-contribution outcomes, so the
        # fallback is not value-changing here; its load-bearing correctness is locked in by the unit test
        # ``test_exact_zero_pivot_fallback_repairs_spurious_zero`` in ``tests/test_utils/test_pfaffian_family.py``.
        state, psi, n = self._deterministic_wire_state(seed, n=3)
        assert not state.degenerate and state.cov.shape[0] >= 2
        mode_index = torch.arange(2 * n)
        cov_t = torch.as_tensor(np.asarray(state.cov))
        gamma_measured = (
            transition_cov(cov_t[:, None], cov_t[None, :]).index_select(-2, mode_index).index_select(-1, mode_index)
        )
        _, log_abs_no_fallback = OutcomeFamily(gamma_measured).all_slog_pfaffians(zero_pivot_fallback=False)
        assert bool(torch.isneginf(log_abs_no_fallback).any())  # exact-zero pivots really occur
        tree = np.asarray(basis_states_probabilities(state.cov, state.weights, list(range(n))))
        np.testing.assert_allclose(
            tree, self._per_outcome_probabilities(state.cov, state.weights, list(range(n))), atol=ATOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(tree, self._statevector_probabilities(psi, n), atol=ATOL_MATRIX_COMPARISON)

    def test_full_distribution_gradient_matches_loop(self):
        # A probability-based loss has matching gradients through the tree and through the per-outcome loop.
        torch.manual_seed(TEST_SEED)
        state, _, n = self._multi_branch_state(1, n=3, n_swaps=2)
        base = torch.as_tensor(np.asarray(state.cov), dtype=torch.float64)  # (chi, D, D)
        weights = torch.as_tensor(np.asarray(state.weights), dtype=torch.complex128)
        measured = list(range(n))
        dim = base.shape[-1]
        coeffs = torch.arange(1, 2**n + 1, dtype=torch.float64)
        binary = NonInteractingFermionicDevice.states_to_binary(np.arange(2**n), n)

        def loss(theta, use_tree):
            rotation = torch.matrix_exp(theta - theta.transpose(-1, -2))  # orthogonal, keeps cov physical
            cov = torch.einsum("ij,bjk,lk->bil", rotation, base, rotation)  # (chi, D, D)
            if use_tree:
                probs = basis_states_probabilities(cov, weights, measured)
            else:
                probs = torch.stack([basis_state_probability(cov, weights, binary[i], measured) for i in range(2**n)])
            return (coeffs * probs).sum()

        theta0 = 0.1 * torch.randn(dim, dim, dtype=torch.float64)
        theta_tree = theta0.clone().requires_grad_(True)
        theta_loop = theta0.clone().requires_grad_(True)
        loss(theta_tree, use_tree=True).backward()
        loss(theta_loop, use_tree=False).backward()
        assert torch.isfinite(theta_tree.grad).all()
        torch.testing.assert_close(
            theta_tree.grad, theta_loop.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
