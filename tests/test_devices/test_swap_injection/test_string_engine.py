import itertools

import numpy as np
import pennylane as qml
import torch

from matchcake import SwapAugmentedFermionicDevice
from matchcake.devices.swap_injection import CzStringEngine, basis_state_probability
from matchcake.operations.state_preparation import ProductState

from ...configs import ATOL_MATRIX_COMPARISON


class TestCzStringEngine:
    """Unit tests for the overlap-free CZ-expansion evaluator.

    Engines are built through ``SwapAugmentedFermionicDevice.string_engine`` (the production
    construction path) and validated against ``default.qubit`` and against the branch-tensor
    observables in the regime where both are exact.
    """

    @staticmethod
    def _device_after(circuit_ops, n):
        device = SwapAugmentedFermionicDevice(wires=n)
        device.apply(circuit_ops)
        return device

    @staticmethod
    def _wire_sharing_ops(x):
        return [
            qml.IsingXX(x[0], wires=[0, 1]),
            qml.IsingYY(x[1], wires=[1, 2]),
            qml.SWAP(wires=[0, 1]),
            qml.IsingXX(x[2], wires=[1, 2]),
            qml.SWAP(wires=[1, 2]),
            qml.IsingYY(x[3], wires=[0, 1]),
            qml.SWAP(wires=[0, 1]),
            qml.IsingXX(x[4], wires=[1, 2]),
        ]

    def test_zero_event_engine_matches_branch_probabilities(self):
        n = 3
        x = np.random.default_rng(0).uniform(-2, 2, size=2)
        device = self._device_after([qml.IsingXX(x[0], wires=[0, 1]), qml.IsingYY(x[1], wires=[1, 2])], n)
        engine = device.string_engine
        branch = device.branch_state
        for bits in itertools.product((0, 1), repeat=n):
            engine_probability = float(engine.basis_state_probability(np.array(bits), list(range(n))))
            branch_probability = float(
                basis_state_probability(branch.cov, branch.weights, np.array(bits), list(range(n)))
            )
            np.testing.assert_allclose(engine_probability, branch_probability, atol=ATOL_MATRIX_COMPARISON)

    def test_single_swap_engine_matches_branch_path(self):
        # Both evaluations are exact for a single SWAP, so they must agree on probs and expvals.
        n = 3
        x = np.random.default_rng(1).uniform(-2, 2, size=2)
        device = self._device_after(
            [qml.IsingXX(x[0], wires=[0, 1]), qml.SWAP(wires=[1, 2]), qml.IsingYY(x[1], wires=[0, 1])], n
        )
        engine = device.string_engine
        branch = device.branch_state
        assert not branch.degenerate
        for bits in itertools.product((0, 1), repeat=n):
            engine_probability = float(engine.basis_state_probability(np.array(bits), list(range(n))))
            branch_probability = float(
                basis_state_probability(branch.cov, branch.weights, np.array(bits), list(range(n)))
            )
            np.testing.assert_allclose(engine_probability, branch_probability, atol=ATOL_MATRIX_COMPARISON)
        observable = qml.Hamiltonian([1.0, 0.5, 0.3], [qml.Z(0) @ qml.X(2), qml.Y(1), qml.Z(2)])
        from matchcake.devices.swap_injection import hamiltonian_expval

        engine_expval = float(engine.hamiltonian_expval(observable, list(range(n))))
        branch_expval = float(
            qml.math.real(
                hamiltonian_expval(branch.cov, branch.weights, observable, list(range(n)), marker=branch.marker)
            )
        )
        np.testing.assert_allclose(engine_expval, branch_expval, atol=ATOL_MATRIX_COMPARISON)

    def test_wire_sharing_marginal_consistent_with_full_sum(self):
        n = 3
        x = np.random.default_rng(2).uniform(-2, 2, size=5)
        device = self._device_after(
            [qml.BasisState(np.array([0, 1, 1]), wires=range(n))] + self._wire_sharing_ops(x), n
        )
        engine = device.string_engine
        full = np.array(
            [
                float(engine.basis_state_probability(np.array(bits), list(range(n))))
                for bits in itertools.product((0, 1), repeat=n)
            ]
        ).reshape([2] * n)
        for bit in (0, 1):
            marginal = float(engine.basis_state_probability(np.array([bit]), [1]))
            np.testing.assert_allclose(marginal, full[:, bit, :].sum(), atol=ATOL_MATRIX_COMPARISON)

    def test_odd_pauli_expval_with_product_state_input(self):
        n = 3
        rng = np.random.default_rng(3)
        amplitudes = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        amplitudes /= np.linalg.norm(amplitudes, axis=1, keepdims=True)
        flat = np.array([1.0 + 0j])
        for k in range(n):
            flat = np.kron(flat, amplitudes[k])
        x = rng.uniform(-2, 2, size=2)
        observable = qml.Hamiltonian([1.0, 0.7], [qml.X(0), qml.Y(2)])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref():
            qml.StatePrep(flat, wires=range(n))
            qml.IsingXX(x[0], wires=[0, 1])
            qml.SWAP(wires=[1, 2])
            qml.IsingYY(x[1], wires=[0, 1])
            return qml.expval(observable)

        device = self._device_after(
            [
                ProductState(amplitudes, wires=range(n)),
                qml.IsingXX(x[0], wires=[0, 1]),
                qml.SWAP(wires=[1, 2]),
                qml.IsingYY(x[1], wires=[0, 1]),
            ],
            n,
        )
        engine_expval = float(device.string_engine.hamiltonian_expval(observable, list(range(n))))
        np.testing.assert_allclose(engine_expval, float(ref()), atol=ATOL_MATRIX_COMPARISON)

    def test_hamiltonian_with_identity_term(self):
        # The identity term exercises the zero-length sandwich bucket (norm <psi|psi> = 1).
        n = 3
        x = np.random.default_rng(4).uniform(-2, 2, size=5)
        observable = qml.Hamiltonian([0.7, 0.5], [qml.Identity(0), qml.PauliZ(1) @ qml.PauliX(2)])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref():
            for op in self._wire_sharing_ops(x):
                qml.apply(op)
            return qml.expval(observable)

        device = self._device_after(self._wire_sharing_ops(x), n)
        engine_expval = float(device.string_engine.hamiltonian_expval(observable, list(range(n))))
        np.testing.assert_allclose(engine_expval, float(ref()), atol=ATOL_MATRIX_COMPARISON)

    def test_bare_pauli_word_observable(self):
        # Exercises the TermsUndefinedError fallback for observables without a terms() split.
        n = 3
        device = self._device_after([qml.IsingXX(0.4, wires=[0, 1]), qml.SWAP(wires=[1, 2])], n)
        engine = device.string_engine
        got = float(engine.hamiltonian_expval(qml.PauliZ(0), list(range(n))))

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref():
            qml.IsingXX(0.4, wires=[0, 1])
            qml.SWAP(wires=[1, 2])
            return qml.expval(qml.PauliZ(0))

        np.testing.assert_allclose(got, float(ref()), atol=ATOL_MATRIX_COMPARISON)

    def test_swaps_without_matchgates(self):
        # No matchgate layer at all: sptm_prefix and sptm_total stay None (identity rotations).
        n = 3
        device = self._device_after(
            [qml.BasisState(np.array([1, 1, 0]), wires=range(n)), qml.SWAP(wires=[0, 1]), qml.SWAP(wires=[1, 2])], n
        )
        engine = device.string_engine

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref():
            qml.BasisState(np.array([1, 1, 0]), wires=range(n))
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[1, 2])
            return qml.probs(wires=range(n))

        expected = np.asarray(ref())
        got = np.array(
            [
                float(engine.basis_state_probability(np.array(bits), list(range(n))))
                for bits in itertools.product((0, 1), repeat=n)
            ]
        )
        np.testing.assert_allclose(got, expected, atol=ATOL_MATRIX_COMPARISON)

    def test_pure_cz_circuit_without_matchgates(self):
        # No matchgate stream at all: the lifted covariance is the input lift and every insertion
        # keeps its identity rotation (sptm_prefix and sptm_total both None).
        n = 3
        device = self._device_after(
            [qml.BasisState(np.array([1, 1, 0]), wires=range(n)), qml.CZ(wires=[0, 1]), qml.CZ(wires=[1, 2])], n
        )
        engine = device.string_engine
        probabilities = np.array(
            [
                float(engine.basis_state_probability(np.array(bits)))  # default measured_qubits
                for bits in itertools.product((0, 1), repeat=n)
            ]
        )
        expected = np.zeros(2**n)
        expected[0b110] = 1.0  # CZ gates only add phases to a basis state
        np.testing.assert_allclose(probabilities, expected, atol=ATOL_MATRIX_COMPARISON)

    def test_direct_construction_with_identity_prefix(self):
        # Constructing the engine directly with an event prefix but no total SPTM exercises the
        # identity-suffix branch; a CZ on |11> only flips the global phase, so p(11) stays 1.
        n = 2
        device = self._device_after([qml.BasisState(np.array([1, 1]), wires=range(n)), qml.CZ(wires=[0, 1])], n)
        lifted_covariance = np.asarray(qml.math.toarray(device._lifted_input_cov))
        engine = CzStringEngine(lifted_covariance, [(0, 1, np.eye(2 * n))], sptm_total=None)
        probability = float(engine.basis_state_probability(np.array([1, 1]), [0, 1]))
        np.testing.assert_allclose(probability, 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_branch_pair_sum_over_all_pairs_reconstructs_expval(self):
        # Completeness identity: sum_{ab} <lambda_a phi_a| H |lambda_b phi_b> over ALL branch pairs
        # equals <H>, evaluated purely from the branch histories with no overlap anywhere.
        n = 3
        x = np.random.default_rng(6).uniform(-2, 2, size=5)
        device = self._device_after(self._wire_sharing_ops(x), n)
        branch = device.branch_state
        observable = qml.Hamiltonian([1.0, 0.5], [qml.Z(0) @ qml.X(2), qml.Y(1)])
        all_pairs = np.ones((branch.chi, branch.chi), dtype=bool)
        got = float(
            device.string_engine.branch_pair_hamiltonian_expval(observable, list(range(n)), branch.histories, all_pairs)
        )

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref():
            for op in self._wire_sharing_ops(x):
                qml.apply(op)
            return qml.expval(observable)

        np.testing.assert_allclose(got, float(ref()), atol=ATOL_MATRIX_COMPARISON)

    def test_branch_pair_cost(self):
        histories = [(0, 0), (1, 0), (1, 1)]
        pair_mask = np.zeros((3, 3), dtype=bool)
        pair_mask[1, 2] = pair_mask[2, 1] = pair_mask[2, 2] = True
        # 4^(1+2) + 4^(2+1) + 4^(2+2) = 64 + 64 + 256
        assert CzStringEngine.branch_pair_cost(histories, pair_mask) == 384

    def test_odd_monomial_vanishes_by_parity(self):
        # A bare (marker-free) odd Majorana monomial has zero expectation in the parity-even
        # lifted algebra; exercises the odd-length guard of the string-pair sum.
        device = self._device_after([qml.IsingXX(0.4, wires=[0, 1]), qml.SWAP(wires=[1, 2])], 3)
        engine = device.string_engine
        value = engine._string_pair_sum_many([(1.0 + 0.0j, [engine._eye[0]])], engine._get_contraction_g())
        np.testing.assert_allclose(complex(value), 0.0, atol=ATOL_MATRIX_COMPARISON)

    def test_batched_sptm_expval(self):
        n = 3
        thetas = torch.linspace(0.2, 1.4, 4, dtype=torch.float64)
        observable = qml.Hamiltonian([1.0, 0.5], [qml.Z(0), qml.Z(2)])
        device = self._device_after(
            [qml.IsingXX(thetas, wires=[0, 1]), qml.SWAP(wires=[1, 2]), qml.IsingYY(0.3, wires=[0, 1])], n
        )
        engine_expval = np.asarray(
            qml.math.toarray(device.string_engine.hamiltonian_expval(observable, list(range(n))))
        )

        expected = []
        for theta in thetas:

            @qml.qnode(qml.device("default.qubit", wires=n))
            def ref():
                qml.IsingXX(float(theta), wires=[0, 1])
                qml.SWAP(wires=[1, 2])
                qml.IsingYY(0.3, wires=[0, 1])
                return qml.expval(observable)

            expected.append(float(ref()))
        np.testing.assert_allclose(engine_expval, np.asarray(expected), atol=ATOL_MATRIX_COMPARISON)
