import itertools

import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.swap_injection import (
    SwapBranchState,
    basis_state_probability,
    hamiltonian_expval,
    lift_from_product_state,
    lift_sptm,
)
from matchcake.devices.swap_injection.lift import DISPLACEMENT_TOL

from ...configs import ATOL_MATRIX_COMPARISON
from . import _oracle as oracle


class TestLift:
    """Even (2n+2) lift -- ports ``verify_displaced_full.py`` and ``verify_edge_cases.py``."""

    @staticmethod
    def _make_prod(amplitudes):
        psi = np.array([1 + 0j])
        for v in amplitudes:
            psi = np.kron(psi, v / np.linalg.norm(v))
        return psi

    @staticmethod
    def _run_displaced(pairs, n, seed):
        """Displaced product-state input through the branch tensor and via statevector."""
        rng = np.random.default_rng(seed)
        cs = oracle.majoranas(n)
        psi_in, _ = oracle.product_state(n, rng)
        covariance, displacement = oracle.phys_cov_disp(psi_in, cs)
        lifted0 = np.asarray(lift_from_product_state(covariance, displacement))
        gates = [oracle.random_gaussian_unitary(n, cs, rng) for _ in range(len(pairs) + 1)]
        sptms = [oracle.sptm_of(g, cs, n) for g in gates]

        psi = gates[0] @ psi_in
        for i, (j, k) in enumerate(pairs):
            psi = gates[i + 1] @ oracle.swap_matrix(n, j, k) @ psi

        state = SwapBranchState(np.stack([lifted0]), np.array([[1.0 + 0j]]), lifted=True)
        state.apply_matchgate_sptm(sptms[0])
        for i, (j, k) in enumerate(pairs):
            state.apply_swap(j, k)
            state.apply_matchgate_sptm(sptms[i + 1])
        return state, psi, cs, rng

    @staticmethod
    def _branch_statevectors(pin, gates, fswaps, num, pairs):
        cur = [gates[0] @ pin]
        for i, (j, k) in enumerate(pairs):
            nxt = []
            for v in cur:
                nxt.append(gates[i + 1] @ fswaps[i] @ v)
                nxt.append(gates[i + 1] @ fswaps[i] @ (-2 * num[j] @ num[k] @ v))
            cur = nxt
        return cur

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_rank_one_ancilla_suffices(self, n):
        rng = np.random.default_rng(n)
        cs = oracle.majoranas(n)
        configs = [
            [rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(n)],  # all superposed
            [(rng.normal(size=2) + 1j * rng.normal(size=2)) if q == 0 else np.array([1.0, 0.0]) for q in range(n)],
            [np.array([1.0, 0]) if rng.random() < 0.5 else np.array([0, 1.0]) for _ in range(n)],  # all basis
        ]
        for amplitudes in configs:
            phi = self._make_prod([np.asarray(a, dtype=complex) for a in amplitudes])
            covariance, _ = oracle.phys_cov_disp(phi, cs)
            rank = np.linalg.matrix_rank(covariance @ covariance + np.eye(2 * n), tol=1e-9)
            assert rank <= 2

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_lift_is_orthogonal_and_preserves_physical_block(self, n):
        rng = np.random.default_rng(10 + n)
        cs = oracle.majoranas(n)
        phi, _ = oracle.product_state(n, rng)
        covariance, displacement = oracle.phys_cov_disp(phi, cs)
        lifted = np.asarray(lift_from_product_state(covariance, displacement))
        assert lifted.shape == (2 * n + 2, 2 * n + 2)
        np.testing.assert_allclose(lifted @ lifted, -np.eye(2 * n + 2), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(lifted[: 2 * n, : 2 * n], covariance, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(lifted[: 2 * n, 2 * n + 1], -displacement, atol=ATOL_MATRIX_COMPARISON)

    def test_disp_zero_reduces_to_physical_block(self):
        n = 3
        cs = oracle.majoranas(n)
        phi = self._make_prod([np.array([1.0, 0]) if b == 0 else np.array([0, 1.0]) for b in [0, 1, 1]])
        covariance, displacement = oracle.phys_cov_disp(phi, cs)
        assert np.linalg.norm(displacement) < DISPLACEMENT_TOL
        lifted = np.asarray(lift_from_product_state(covariance, displacement))
        np.testing.assert_allclose(lifted[: 2 * n, : 2 * n], covariance, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(lifted[: 2 * n, 2 * n :], 0.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(lifted[2 * n, 2 * n + 1], 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_lift_sptm_is_q_plus_identity(self):
        rng = np.random.default_rng(2)
        n = 3
        cs = oracle.majoranas(n)
        sptm = oracle.sptm_of(oracle.random_gaussian_unitary(n, cs, rng), cs, n)
        lifted = np.asarray(lift_sptm(sptm))
        np.testing.assert_allclose(lifted[: 2 * n, : 2 * n], sptm, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(lifted[2 * n :, 2 * n :], np.eye(2), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(lifted[: 2 * n, 2 * n :], 0.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize(
        "pairs, n, seed, expected_chi",
        [
            ([(0, 1)], 3, 31, 2),
            ([(0, 1), (2, 3)], 4, 99, 4),
            ([(0, 1), (1, 2), (2, 3)], 4, 7, 8),
        ],
    )
    def test_displaced_probabilities_match_and_sum_to_one(self, pairs, n, seed, expected_chi):
        state, psi, _, _ = self._run_displaced(pairs, n, seed)
        assert state.chi == expected_chi
        total = 0.0
        for y in itertools.product((0, 1), repeat=n):
            got = float(basis_state_probability(state.cov, state.weights, np.array(y)))
            total += got
            np.testing.assert_allclose(
                got, abs(oracle.basis_state(n, y).conj() @ psi) ** 2, atol=ATOL_MATRIX_COMPARISON
            )
        np.testing.assert_allclose(total, 1.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("pairs, n, seed", [([(0, 1)], 3, 31), ([(0, 1), (1, 2), (2, 3)], 4, 7)])
    def test_displaced_hamiltonian_with_parity_odd_terms(self, pairs, n, seed):
        state, psi, cs, rng = self._run_displaced(pairs, n, seed)
        terms = [(rng.normal(), "".join(rng.choice(list("IXYZ"), size=n))) for _ in range(10)]
        # Ensure at least one parity-odd term is present (these vanish on the basis path).
        has_odd = any(len(oracle.pauli_to_majorana_bruteforce(p, n, cs)[0]) % 2 == 1 for _, p in terms)
        assert has_odd, "expected a parity-odd Pauli term to exercise the marker append"
        hamiltonian = qml.Hamiltonian([c for c, _ in terms], [qml.pauli.string_to_pauli_word(p) for _, p in terms])
        dense = sum(c * oracle.kron_list([oracle.PAULI_MAP[ch] for ch in p]) for c, p in terms)
        exact = (psi.conj() @ dense @ psi).real
        got = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=state.marker))
        np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_propagate_passes_and_rebuild_fails_seed55(self):
        """Propagate-the-lift is exact; per-branch rebuild fails (verify_edge_cases test 5).

        The branch covariances must come from propagating one shared ancilla frame, never from
        rebuilding each branch's lift from its own (cov, disp). Rebuild picks inconsistent frames
        and corrupts the projector inverse, so even p(y) (which reads only the physical block) is
        wrong. A single green test does not prove correctness (rebuild accidentally passes on some
        circuits), so this pins the seed-55 circuit where the failure is decisive.
        """
        rng = np.random.default_rng(55)
        n = 4
        cs = oracle.majoranas(n)
        num = oracle.number_ops(n, cs)
        pairs = [(0, 1), (2, 3)]
        pin, _ = oracle.product_state(n, rng)
        lifted0 = np.asarray(lift_from_product_state(*oracle.phys_cov_disp(pin, cs)))
        gates = [oracle.random_gaussian_unitary(n, cs, rng) for _ in range(3)]
        sptms = [oracle.sptm_of(g, cs, n) for g in gates]
        fswaps = [oracle.fswap_matrix(n, j, k, num) for (j, k) in pairs]

        psi = gates[0] @ pin
        for i, (j, k) in enumerate(pairs):
            psi = gates[i + 1] @ oracle.swap_matrix(n, j, k) @ psi

        state = SwapBranchState(np.stack([lifted0]), np.array([[1.0 + 0j]]), lifted=True)
        state.apply_matchgate_sptm(sptms[0])
        for i, (j, k) in enumerate(pairs):
            state.apply_swap(j, k)
            state.apply_matchgate_sptm(sptms[i + 1])

        vectors = self._branch_statevectors(pin, gates, fswaps, num, pairs)
        weights = np.array([[vectors[a].conj() @ vectors[b] for b in range(len(vectors))] for a in range(len(vectors))])
        rebuilt = np.stack(
            [np.asarray(lift_from_product_state(*oracle.phys_cov_disp(v / np.linalg.norm(v), cs))) for v in vectors]
        )

        def max_prob_error(covariances, weight_matrix):
            error = 0.0
            for y in itertools.product((0, 1), repeat=n):
                got = float(basis_state_probability(covariances, weight_matrix, np.array(y)))
                error = max(error, abs(got - abs(oracle.basis_state(n, y).conj() @ psi) ** 2))
            return error

        error_propagate = max_prob_error(state.cov, state.weights)
        error_rebuild = max_prob_error(rebuilt, weights)
        assert error_propagate < 1e-7, f"propagation should be exact, got {error_propagate}"
        assert error_rebuild > 1e-3, f"per-branch rebuild must fail on seed 55, got {error_rebuild}"
