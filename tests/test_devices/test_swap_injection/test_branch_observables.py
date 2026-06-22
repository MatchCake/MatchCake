import itertools

import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.probability_strategies.product_state_strategy import (
    ProductStateProbabilityStrategy,
)
from matchcake.devices.swap_injection import (
    SwapBranchState,
    basis_state_probability,
    hamiltonian_expval,
    transition_cov,
)

from ...configs import ATOL_MATRIX_COMPARISON
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
