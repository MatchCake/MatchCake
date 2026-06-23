import itertools

import numpy as np
import pennylane as qml

from matchcake.devices.probability_strategies.product_state_strategy import (
    ProductStateProbabilityStrategy,
)
from matchcake.devices.swap_injection import (
    SwapBranchState,
    basis_state_probability,
    condition_occupied,
    hamiltonian_expval,
)

from ...configs import ATOL_MATRIX_COMPARISON
from . import _oracle as oracle


class TestBranchState:
    """SwapBranchState container + condition_occupied -- ports ``verify_update_rules.py``."""

    @staticmethod
    def _two_swap_pipeline(rng):
        """G3 . SWAP(2,3) . G2 . SWAP(0,1) . G1 |1100>, n=4 -- the verify_update_rules circuit."""
        n = 4
        cs = oracle.majoranas(n)
        x0 = [1, 1, 0, 0]
        gates = [oracle.random_gaussian_unitary(n, cs, rng) for _ in range(3)]
        sptms = [oracle.sptm_of(g, cs, n) for g in gates]
        psi = (
            gates[2]
            @ oracle.swap_matrix(n, 2, 3)
            @ gates[1]
            @ oracle.swap_matrix(n, 0, 1)
            @ gates[0]
            @ oracle.basis_state(n, x0)
        )
        lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)
        state = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        state.apply_matchgate_sptm(sptms[0])
        state.apply_swap(0, 1)
        state.apply_matchgate_sptm(sptms[1])
        state.apply_swap(2, 3)
        state.apply_matchgate_sptm(sptms[2])
        return state, psi, n

    @staticmethod
    def _two_cz_pipeline(rng):
        """G3 . CZ(2,3) . G2 . CZ(0,1) . G1 |1100>, n=4 -- a SWAP pipeline with the fSWAP dropped."""
        n = 4
        cs = oracle.majoranas(n)
        num = oracle.number_ops(n, cs)
        x0 = [1, 1, 0, 0]
        gates = [oracle.random_gaussian_unitary(n, cs, rng) for _ in range(3)]
        sptms = [oracle.sptm_of(g, cs, n) for g in gates]

        def cz(j, k):
            return np.eye(2**n) - 2 * num[j] @ num[k]

        psi = gates[2] @ cz(2, 3) @ gates[1] @ cz(0, 1) @ gates[0] @ oracle.basis_state(n, x0)
        lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)
        state = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        state.apply_matchgate_sptm(sptms[0])
        state.apply_cz(0, 1)
        state.apply_matchgate_sptm(sptms[1])
        state.apply_cz(2, 3)
        state.apply_matchgate_sptm(sptms[2])
        return state, psi, n

    def test_cz_probabilities_match_statevector(self):
        rng = np.random.default_rng(11)
        state, psi, n = self._two_cz_pipeline(rng)
        for y in itertools.product((0, 1), repeat=n):
            got = float(basis_state_probability(state.cov, state.weights, np.array(y)))
            exact = abs(oracle.basis_state(n, y).conj() @ psi) ** 2
            np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_cz_weights_sum_to_one(self):
        rng = np.random.default_rng(11)
        state, _, _ = self._two_cz_pipeline(rng)
        np.testing.assert_allclose(complex(np.sum(np.asarray(state.weights))).real, 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_cz_chi_doubles_per_branching(self):
        rng = np.random.default_rng(11)
        state, _, _ = self._two_cz_pipeline(rng)
        assert state.chi == 4  # two CZ branchings on initially-occupied modes -> chi = 2^2

    def test_cz_hamiltonian_matches_statevector(self):
        rng = np.random.default_rng(11)
        state, psi, n = self._two_cz_pipeline(rng)
        terms = [(rng.normal(), "".join(rng.choice(list("IXYZ"), size=n))) for _ in range(4)]
        hamiltonian = qml.Hamiltonian([c for c, _ in terms], [qml.pauli.string_to_pauli_word(p) for _, p in terms])
        dense = sum(c * oracle.kron_list([oracle.PAULI_MAP[ch] for ch in p]) for c, p in terms)
        exact = (psi.conj() @ dense @ psi).real
        got = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=None))
        np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_cz_then_fswap_equals_swap(self):
        # SWAP = fSWAP . CZ, so apply_cz followed by the fSWAP matchgate must equal apply_swap.
        rng = np.random.default_rng(7)
        n = 3
        cs = oracle.majoranas(n)
        x0 = [1, 1, 0]
        sptm = oracle.sptm_of(oracle.random_gaussian_unitary(n, cs, rng), cs, n)
        lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)

        from pennylane.wires import Wires

        from matchcake.operations.single_particle_transition_matrices.sptm_fswap import SptmCompZX

        swapped = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        swapped.apply_matchgate_sptm(sptm)
        swapped.apply_swap(0, 1)

        cz_then_fswap = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        cz_then_fswap.apply_matchgate_sptm(sptm)
        cz_then_fswap.apply_cz(0, 1)
        fswap_sptm = SptmCompZX(wires=[0, 1]).pad(Wires(range(n))).matrix()
        cz_then_fswap.apply_matchgate_sptm(fswap_sptm)

        np.testing.assert_allclose(np.asarray(cz_then_fswap.cov), np.asarray(swapped.cov), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(
            np.asarray(cz_then_fswap.weights), np.asarray(swapped.weights), atol=ATOL_MATRIX_COMPARISON
        )

    def test_condition_occupied_matches_projected_state(self):
        rng = np.random.default_rng(0)
        n = 4
        cs = oracle.majoranas(n)
        num = oracle.number_ops(n, cs)
        j, k = 0, 2
        phi = oracle.random_gaussian_unitary(n, cs, rng) @ oracle.basis_state(n, [1, 1, 0, 0])
        covariance, _ = oracle.phys_cov_disp(phi, cs)
        projected = num[j] @ num[k] @ phi
        projected = projected / np.linalg.norm(projected)
        covariance_true, _ = oracle.phys_cov_disp(projected, cs)
        got = np.asarray(condition_occupied(covariance, j, k))
        np.testing.assert_allclose(got, covariance_true, atol=ATOL_MATRIX_COMPARISON)

    def test_condition_occupied_accepts_integer_input(self):
        # An integer-typed covariance (e.g. a basis-state Lambda_y) is cast to float internally.
        n = 3
        lambda_y = ProductStateProbabilityStrategy.build_lambda_y(np.array([1, 1, 0]), n).astype(int)
        got = np.asarray(condition_occupied(lambda_y, 0, 1))
        reference = np.asarray(condition_occupied(lambda_y.astype(float), 0, 1))
        np.testing.assert_allclose(got, reference, atol=ATOL_MATRIX_COMPARISON)

    def test_condition_occupied_pins_block_and_zeros_offblock(self):
        rng = np.random.default_rng(5)
        n = 3
        cs = oracle.majoranas(n)
        phi = oracle.random_gaussian_unitary(n, cs, rng) @ oracle.basis_state(n, [1, 1, 0])
        covariance, _ = oracle.phys_cov_disp(phi, cs)
        out = np.asarray(condition_occupied(covariance, 0, 1))
        occupied = [0, 1, 2, 3]
        occupied_block = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
        np.testing.assert_allclose(out[np.ix_(occupied, occupied)], occupied_block, atol=ATOL_MATRIX_COMPARISON)
        rest = [m for m in range(2 * n) if m not in occupied]
        np.testing.assert_allclose(out[np.ix_(occupied, rest)], 0.0, atol=ATOL_MATRIX_COMPARISON)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(11)
        state, _, _ = self._two_swap_pipeline(rng)
        np.testing.assert_allclose(complex(np.sum(np.asarray(state.weights))).real, 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_chi_doubles_per_full_swap(self):
        rng = np.random.default_rng(11)
        state, _, _ = self._two_swap_pipeline(rng)
        assert state.chi == 4  # two genuine SWAPs on initially-occupied modes -> chi = 2^2

    def test_total_probability_equals_full_weight_sum(self):
        # sum_y p(y) = <psi|psi> = sum_{a,b} W_{ab} (the full weight sum, not just the diagonal: the
        # off-diagonal branch interference contributes to the norm).
        rng = np.random.default_rng(11)
        state, _, n = self._two_swap_pipeline(rng)
        weight_sum = complex(np.sum(np.asarray(state.weights))).real
        prob_sum = sum(
            float(basis_state_probability(state.cov, state.weights, np.array(y)))
            for y in itertools.product((0, 1), repeat=n)
        )
        np.testing.assert_allclose(prob_sum, weight_sum, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(prob_sum, 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_probabilities_match_statevector(self):
        rng = np.random.default_rng(11)
        state, psi, n = self._two_swap_pipeline(rng)
        for y in itertools.product((0, 1), repeat=n):
            got = float(basis_state_probability(state.cov, state.weights, np.array(y)))
            exact = abs(oracle.basis_state(n, y).conj() @ psi) ** 2
            np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_hamiltonian_matches_statevector(self):
        rng = np.random.default_rng(11)
        state, psi, n = self._two_swap_pipeline(rng)
        terms = [(rng.normal(), "".join(rng.choice(list("IXYZ"), size=n))) for _ in range(4)]
        hamiltonian = qml.Hamiltonian([c for c, _ in terms], [qml.pauli.string_to_pauli_word(p) for _, p in terms])
        dense = sum(c * oracle.kron_list([oracle.PAULI_MAP[ch] for ch in p]) for c, p in terms)
        exact = (psi.conj() @ dense @ psi).real
        got = float(hamiltonian_expval(state.cov, state.weights, hamiltonian, list(range(n)), marker=None))
        np.testing.assert_allclose(got, exact, atol=ATOL_MATRIX_COMPARISON)

    def test_pruning_removes_vanished_ancilla_branch(self):
        n = 3
        # qubit 2 stays in |0>; SWAP(0, 2) projects onto n_0 n_2 = 0 -> type-1 branch vanishes.
        x0 = [1, 0, 0]
        lambda0 = ProductStateProbabilityStrategy.build_lambda_y(np.array(x0), n).astype(float)
        state = SwapBranchState(np.stack([lambda0]), np.array([[1.0 + 0j]]), lifted=False)
        state.apply_swap(0, 2)
        assert state.chi == 1  # the occupied-projection branch had |W| = 0 and was pruned
