import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.exceptions import DeviceError

from matchcake import NonInteractingFermionicDevice, SwapAugmentedFermionicDevice
from matchcake.operations.state_preparation import ProductState

from ...configs import ATOL_MATRIX_COMPARISON


class TestSwapAugmentedFermionicDevice:
    """End-to-end device tests through the ``qml`` API.

    Covered: zero-SWAP (strict superset of ``nif.qubit``), a single SWAP, SWAPs on disjoint wire
    pairs, and wire-sharing SWAPs (rerouted to the overlap-free string engine, see
    ``swap_injection_theory.md`` section 11), for basis and product-state inputs,
    ``probs``/``expval``/marginals/batched params.
    """

    OBS = qml.PauliZ(0) @ qml.PauliX(2) + 0.5 * qml.PauliY(1) + 0.3 * qml.PauliX(0)

    @staticmethod
    def _zero_swap(x):
        qml.IsingXX(x[0], wires=[0, 1])
        qml.IsingYY(x[1], wires=[1, 2])
        qml.IsingXX(x[2], wires=[0, 1])

    @staticmethod
    def _single_swap(x):
        qml.IsingXX(x[0], wires=[0, 1])
        qml.SWAP(wires=[1, 2])
        qml.IsingYY(x[1], wires=[0, 1])
        qml.IsingXX(x[2], wires=[1, 2])

    @staticmethod
    def _disjoint_swaps(x):
        qml.IsingXX(x[0], wires=[0, 1])
        qml.IsingYY(x[1], wires=[2, 3])
        qml.SWAP(wires=[0, 1])
        qml.IsingXX(x[2], wires=[2, 3])
        qml.SWAP(wires=[2, 3])
        qml.IsingYY(x[3], wires=[0, 1])

    @staticmethod
    def _wire_sharing_swaps(x):
        qml.IsingXX(x[0], wires=[0, 1])
        qml.IsingYY(x[1], wires=[1, 2])
        qml.SWAP(wires=[0, 1])
        qml.IsingXX(x[2], wires=[1, 2])
        qml.SWAP(wires=[1, 2])
        qml.IsingYY(x[3], wires=[0, 1])
        qml.SWAP(wires=[0, 1])
        qml.IsingXX(x[4], wires=[1, 2])

    @staticmethod
    def _single_cz(x):
        qml.IsingXX(x[0], wires=[0, 1])
        qml.CZ(wires=[1, 2])
        qml.IsingYY(x[1], wires=[0, 1])
        qml.IsingXX(x[2], wires=[1, 2])

    @staticmethod
    def _cz_and_swap(x):
        qml.IsingXX(x[0], wires=[0, 1])
        qml.CZ(wires=[0, 1])
        qml.SWAP(wires=[1, 2])
        qml.IsingYY(x[1], wires=[0, 1])

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_cz_probs_match_default_qubit(self, seed):
        n = 3
        x = np.random.default_rng(seed).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_cz(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_cz(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_single_cz_expval_match_default_qubit(self):
        n = 3
        x = np.random.default_rng(3).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_cz(x)
            return qml.expval(self.OBS)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_cz(x)
            return qml.expval(self.OBS)

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_cz_and_swap_probs_match_default_qubit(self):
        n = 3
        x = np.random.default_rng(4).uniform(-2, 2, size=2)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._cz_and_swap(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._cz_and_swap(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_cz_with_rz_match_default_qubit(self):
        # CZ branching composed with single-qubit R_Z (Gaussian) layers.
        n = 3
        x = np.random.default_rng(5).uniform(-2, 2, size=4)

        def body(x):
            qml.RZ(x[0], wires=0)
            qml.IsingXX(x[1], wires=[0, 1])
            qml.CZ(wires=[0, 1])
            qml.RZ(x[2], wires=2)
            qml.IsingYY(x[3], wires=[1, 2])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            body(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            body(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_cz_is_listed_as_supported(self):
        assert "CZ" in SwapAugmentedFermionicDevice._supported_ops

    def test_zero_swap_matches_nif_exactly(self):
        n = 3
        x = np.random.default_rng(0).uniform(-2, 2, size=3)
        nif = NonInteractingFermionicDevice(wires=n)
        swap = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(nif)
        def nif_probs(x):
            self._zero_swap(x)
            return qml.probs(wires=range(n))

        @qml.qnode(swap)
        def swap_probs(x):
            self._zero_swap(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(swap_probs(x)), np.asarray(nif_probs(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_zero_swap_expval_matches_default_qubit(self):
        n = 3
        x = np.random.default_rng(1).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._zero_swap(x)
            return qml.expval(self.OBS)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._zero_swap(x)
            return qml.expval(self.OBS)

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_swap_probs_match_default_qubit(self, seed):
        n = 3
        x = np.random.default_rng(seed).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_single_swap_expval_matches_default_qubit(self):
        n = 3
        x = np.random.default_rng(3).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_swap(x)
            return qml.expval(self.OBS)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_swap(x)
            return qml.expval(self.OBS)

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_disjoint_swaps_probs_match_default_qubit(self):
        n = 4
        x = np.random.default_rng(4).uniform(-2, 2, size=4)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._disjoint_swaps(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._disjoint_swaps(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_disjoint_swaps_marginal_probs_match_default_qubit(self):
        n = 4
        x = np.random.default_rng(5).uniform(-2, 2, size=4)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._disjoint_swaps(x)
            return qml.probs(wires=[0, 2])

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._disjoint_swaps(x)
            return qml.probs(wires=[0, 2])

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_disjoint_swaps_batched_params_match_default_qubit(self):
        n = 4
        x = np.random.default_rng(6).uniform(-2, 2, size=(5, 4))

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._disjoint_swaps(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._disjoint_swaps(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("seed", [0, 5, 17])
    def test_wire_sharing_swaps_probs_match_default_qubit(self, seed):
        n = 3
        x = np.random.default_rng(seed).uniform(-2, 2, size=5)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_wire_sharing_swaps_expval_match_default_qubit(self):
        n = 3
        x = np.random.default_rng(17).uniform(-2, 2, size=5)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._wire_sharing_swaps(x)
            return qml.expval(self.OBS)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.expval(self.OBS)

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_wire_sharing_swaps_marginal_probs_match_default_qubit(self):
        n = 3
        x = np.random.default_rng(18).uniform(-2, 2, size=5)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=[0, 2])

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=[0, 2])

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_wire_sharing_swaps_product_state_input(self):
        n = 3
        rng = np.random.default_rng(19)
        amps = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        amps /= np.linalg.norm(amps, axis=1, keepdims=True)
        flat = np.array([1.0 + 0j])
        for k in range(n):
            flat = np.kron(flat, amps[k])
        x = rng.uniform(-2, 2, size=5)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            qml.StatePrep(flat, wires=range(n))
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            ProductState(amps, wires=range(n))
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_wire_sharing_swaps_set_degenerate_flag(self):
        # Wire-sharing SWAPs drive branch pairs exactly orthogonal; the device must detect it and
        # route observables through the overlap-free string engine.
        n = 3
        x = np.random.default_rng(17).uniform(-2, 2, size=5)
        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        got(x)
        assert dev.branch_state.degenerate
        assert dev._string_engine is not None  # the engine was actually built and used

    @pytest.mark.parametrize("swap_wires", [[0, 2], [2, 0], [0, 3]])
    def test_non_adjacent_swap_matches_default_qubit(self, swap_wires):
        # SWAP_{jk} = M_{jk} CZ_{jk} prod_{j<l<k} CZ_{jl} CZ_{kl}: the crossed-mode parity phases
        # are genuine CZ branchings; without them the probabilities were off by O(1e-2).
        n = 4
        x = np.random.default_rng(22).uniform(-2, 2, size=3)

        def body(x):
            qml.IsingXX(x[0], wires=[0, 1])
            qml.IsingYY(x[1], wires=[1, 2])
            qml.SWAP(wires=swap_wires)
            qml.IsingYY(x[2], wires=[0, 1])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref_probs(x):
            body(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got_probs(x):
            body(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got_probs(x)), np.asarray(ref_probs(x)), atol=ATOL_MATRIX_COMPARISON)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref_expval(x):
            body(x)
            return qml.expval(self.OBS)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got_expval(x):
            body(x)
            return qml.expval(self.OBS)

        np.testing.assert_allclose(float(got_expval(x)), float(ref_expval(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_batched_params_with_orthogonal_elements(self):
        # theta = [0, pi] makes the two batch elements nearly orthogonal states; the greedy
        # reference basis state must be chosen per element, not from element 0 only. The leading
        # SWAP has no matchgate prefix, exercising the unbatched pass-through of the per-element
        # slicing.
        n = 3
        theta = np.array([0.0, np.pi])

        def body(theta):
            qml.SWAP(wires=[1, 2])
            qml.IsingXX(theta, wires=[0, 1])
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[1, 2])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(theta):
            qml.BasisState(np.array([1, 1, 0]), wires=range(n))
            body(theta)
            return qml.probs(wires=range(n))

        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def got(theta):
            qml.BasisState(np.array([1, 1, 0]), wires=range(n))
            body(theta)
            return qml.probs(wires=range(n))

        got_probs = np.asarray(got(theta))
        assert dev.branch_state.degenerate  # the fix under test is the engine's per-element reference
        np.testing.assert_allclose(got_probs, np.asarray(ref(theta)), atol=ATOL_MATRIX_COMPARISON)

    def test_hybrid_matches_full_engine_and_default_qubit(self):
        # On a wire-sharing circuit with a partial pair mask, the hybrid per-pair evaluation (the
        # device path), the full CZ-expansion engine, and default.qubit must all agree.
        n = 3
        x = np.random.default_rng(23).uniform(-2, 2, size=5)
        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.expval(self.OBS)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._wire_sharing_swaps(x)
            return qml.expval(self.OBS)

        hybrid_value = float(got(x))
        branch = dev.branch_state
        mask = branch.string_pair_mask
        assert mask.any() and not mask.all()  # genuinely partial: some pairs healthy, some masked
        assert not dev._prefer_full_engine(mask)  # the device path under test is the hybrid one
        engine_value = float(dev.string_engine.hamiltonian_expval(self.OBS, list(range(n))))
        reference = float(ref(x))
        np.testing.assert_allclose(hybrid_value, reference, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(engine_value, reference, atol=ATOL_MATRIX_COMPARISON)

    def test_forced_full_mask_prefers_engine_and_stays_exact(self):
        # With every pair masked by force, the cost heuristic must hand the whole evaluation to the
        # full engine, and the results must stay exact.
        n = 3
        x = np.random.default_rng(24).uniform(-2, 2, size=2)
        dev = SwapAugmentedFermionicDevice(wires=n)
        dev.apply(
            [
                qml.BasisState(np.array([0, 1, 1]), wires=range(n)),
                qml.IsingXX(x[0], wires=[0, 1]),
                qml.SWAP(wires=[1, 2]),
                qml.IsingYY(x[1], wires=[0, 1]),
            ]
        )
        branch = dev.branch_state
        branch._pair_needs_string[:] = True
        assert dev._prefer_full_engine(branch.string_pair_mask)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            qml.BasisState(np.array([0, 1, 1]), wires=range(n))
            qml.IsingXX(x[0], wires=[0, 1])
            qml.SWAP(wires=[1, 2])
            qml.IsingYY(x[1], wires=[0, 1])
            return qml.probs(wires=[0, 2])

        marginal = np.asarray(
            qml.math.toarray(
                qml.math.stack(
                    [dev._degenerate_probability(np.array(bits), [0, 2]) for bits in ([0, 0], [0, 1], [1, 0], [1, 1])]
                )
            )
        )
        np.testing.assert_allclose(marginal, np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)
        expval = float(dev._degenerate_expval(self.OBS))

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref_expval(x):
            qml.BasisState(np.array([0, 1, 1]), wires=range(n))
            qml.IsingXX(x[0], wires=[0, 1])
            qml.SWAP(wires=[1, 2])
            qml.IsingYY(x[1], wires=[0, 1])
            return qml.expval(self.OBS)

        np.testing.assert_allclose(expval, float(ref_expval(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_wire_sharing_swaps_single_target_probability(self):
        # Exercises the single-outcome degenerate path of get_states_probability.
        n = 3
        x = np.random.default_rng(21).uniform(-2, 2, size=5)
        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def probs(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        full = np.asarray(probs(x))
        assert dev.branch_state.degenerate
        for outcome in range(2**n):
            np.testing.assert_allclose(
                float(dev.get_states_probability(outcome)), full[outcome], atol=ATOL_MATRIX_COMPARISON
            )

    def test_wire_sharing_swaps_batched_params_match_default_qubit(self):
        n = 3
        x = np.random.default_rng(20).uniform(-2, 2, size=(5, 3))

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_product_state_input_disjoint_swaps(self):
        n = 3
        rng = np.random.default_rng(7)
        amps = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        amps /= np.linalg.norm(amps, axis=1, keepdims=True)
        flat = np.array([1.0 + 0j])
        for k in range(n):
            flat = np.kron(flat, amps[k])
        x = rng.uniform(-2, 2, size=2)

        def body(x):
            qml.IsingXX(x[0], wires=[0, 1])
            qml.SWAP(wires=[0, 1])
            qml.IsingYY(x[1], wires=[1, 2])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            qml.StatePrep(flat, wires=range(n))
            body(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            ProductState(amps, wires=range(n))
            body(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_sampling_approximates_analytic(self):
        n = 3
        x = np.random.default_rng(8).uniform(-2, 2, size=3)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def analytic(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n, shots=20000))
        def sampled(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(sampled(x)), np.asarray(analytic(x)), atol=2e-2)

    def test_branch_covariances_shape(self):
        n = 3
        swap = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(swap)
        def circuit():
            qml.IsingXX(0.3, wires=[0, 1])
            qml.SWAP(wires=[1, 2])
            return qml.probs(wires=range(n))

        circuit()
        covariances = np.asarray(swap.branch_covariances)
        assert covariances.shape[-1] == 2 * n + 2  # lifted (2n+2) frame
        assert covariances.ndim >= 3  # (chi, ..., D, D)

    def test_get_states_probability_accepts_int_and_str(self):
        # Single-outcome path + the int/str normalization branches.
        n = 3
        x = np.random.default_rng(10).uniform(-2, 2, size=3)
        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def probs(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        full = np.asarray(probs(x))
        for outcome in range(2**n):
            bits = [int(b) for b in format(outcome, f"0{n}b")]
            from_int = float(dev.get_states_probability(outcome))
            from_str = float(dev.get_states_probability("".join(map(str, bits))))
            np.testing.assert_allclose(from_int, full[outcome], atol=ATOL_MATRIX_COMPARISON)
            np.testing.assert_allclose(from_str, full[outcome], atol=ATOL_MATRIX_COMPARISON)

    def test_identity_operation_is_skipped(self):
        n = 3
        x = np.random.default_rng(11).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            qml.Identity(wires=0)
            self._single_swap(x)
            qml.Identity(wires=1)
            return qml.probs(wires=range(n))

        np.testing.assert_allclose(np.asarray(got(x)), np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_basis_state_projector_expval(self):
        # Routes through exact_expval's projector branch and the single-outcome probability path.
        n = 3
        x = np.random.default_rng(12).uniform(-2, 2, size=3)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_swap(x)
            return qml.expval(qml.Projector(np.array([0, 1, 1]), wires=range(n)))

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_swap(x)
            return qml.expval(qml.Projector(np.array([0, 1, 1]), wires=range(n)))

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_hamiltonian_with_identity_term(self):
        # Exercises the rank-0 (identity) Pauli term branch in hamiltonian_expval.
        n = 3
        x = np.random.default_rng(13).uniform(-2, 2, size=3)
        observable = qml.Hamiltonian([0.7, 0.5], [qml.Identity(0), qml.PauliZ(1) @ qml.PauliX(2)])

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_swap(x)
            return qml.expval(observable)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_swap(x)
            return qml.expval(observable)

        np.testing.assert_allclose(float(got(x)), float(ref(x)), atol=ATOL_MATRIX_COMPARISON)

    def test_invalid_state_prep_raises(self):
        dev = SwapAugmentedFermionicDevice(wires=3)
        dev._state_prep_op = qml.StatePrep(np.eye(8)[0], wires=range(3))
        with pytest.raises(DeviceError):
            _ = dev.branch_state

    def test_wires_inferred_from_operations(self):
        # Device constructed without wires infers them from the applied operations.
        dev_inferred = SwapAugmentedFermionicDevice()
        dev_inferred.apply([qml.IsingXX(0.4, wires=[0, 1]), qml.SWAP(wires=[1, 2]), qml.IsingYY(0.2, wires=[0, 1])])
        dev_fixed = SwapAugmentedFermionicDevice(wires=3)
        dev_fixed.apply([qml.IsingXX(0.4, wires=[0, 1]), qml.SWAP(wires=[1, 2]), qml.IsingYY(0.2, wires=[0, 1])])
        assert dev_inferred.num_wires == 3
        np.testing.assert_allclose(
            np.asarray(dev_inferred.analytic_probability()),
            np.asarray(dev_fixed.analytic_probability()),
            atol=ATOL_MATRIX_COMPARISON,
        )

    def test_identity_operation_applied_directly_is_skipped(self):
        # Applied directly (not via preprocessing), an Identity hits the explicit skip in apply_generator.
        n = 3
        dev_with = SwapAugmentedFermionicDevice(wires=n)
        dev_with.apply([qml.Identity(wires=0), qml.IsingXX(0.5, wires=[0, 1]), qml.SWAP(wires=[1, 2])])
        dev_without = SwapAugmentedFermionicDevice(wires=n)
        dev_without.apply([qml.IsingXX(0.5, wires=[0, 1]), qml.SWAP(wires=[1, 2])])
        np.testing.assert_allclose(
            np.asarray(dev_with.analytic_probability()),
            np.asarray(dev_without.analytic_probability()),
            atol=ATOL_MATRIX_COMPARISON,
        )

    def test_basis_state_prep_op_is_promoted(self):
        # A BasisState left as the state-prep op is promoted to a ProductState inside the branch build.
        n = 3
        dev = SwapAugmentedFermionicDevice(wires=n)
        dev.apply([qml.IsingXX(0.3, wires=[0, 1]), qml.SWAP(wires=[1, 2])])
        dev._state_prep_op = qml.BasisState(np.array([0, 1, 1]), wires=range(n))
        dev._branch_state = None
        assert dev.branch_state.chi >= 1
        probs = np.asarray(dev.analytic_probability())
        np.testing.assert_allclose(probs.sum(), 1.0, atol=ATOL_MATRIX_COMPARISON)

    @staticmethod
    def _swap_expval_torch(theta):
        # Lower-level device API keeps the torch graph intact across the SWAP branching, unlike the
        # qnode boundary. A genuine SWAP sits between the two trainable matchgate layers so the
        # gradient is forced through SwapBranchState.apply_swap.
        n = 4
        dev = SwapAugmentedFermionicDevice(wires=n)

        def generator():
            yield qml.BasisState(np.zeros(n, dtype=int), wires=range(n))
            yield qml.IsingXX(theta, wires=[0, 1])
            yield qml.SWAP(wires=[1, 2])
            yield qml.IsingYY(theta, wires=[2, 3])

        dev.execute_generator(generator(), reset=True, n_ops=8, gc_op=True)
        observable = qml.Hamiltonian([1.0, 0.5], [qml.Z(2), qml.Z(0)])
        return dev.exact_expval(observable).real

    def test_swap_gradient_matches_finite_difference(self):
        theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        expectation = self._swap_expval_torch(theta)
        expectation.backward()
        grad_autograd = theta.grad

        assert torch.isfinite(grad_autograd)
        step = 1e-5
        with torch.no_grad():
            plus = self._swap_expval_torch(torch.tensor(0.5 + step, dtype=torch.float64))
            minus = self._swap_expval_torch(torch.tensor(0.5 - step, dtype=torch.float64))
        grad_finite_difference = (plus - minus) / (2 * step)
        torch.testing.assert_close(grad_autograd, grad_finite_difference, atol=ATOL_MATRIX_COMPARISON, rtol=0.0)

    @staticmethod
    def _wire_sharing_expval_torch(theta):
        # Same lower-level pattern as _swap_expval_torch, but with wire-sharing SWAPs so the
        # gradient is forced through the degenerate path (the CzStringEngine evaluation).
        n = 3
        dev = SwapAugmentedFermionicDevice(wires=n)

        def generator():
            yield qml.BasisState(np.array([0, 1, 1]), wires=range(n))
            yield qml.IsingXX(theta, wires=[0, 1])
            yield qml.IsingYY(0.7, wires=[1, 2])
            yield qml.SWAP(wires=[0, 1])
            yield qml.IsingXX(theta, wires=[1, 2])
            yield qml.SWAP(wires=[1, 2])
            yield qml.IsingYY(0.9, wires=[0, 1])
            yield qml.SWAP(wires=[0, 1])

        dev.execute_generator(generator(), reset=True, n_ops=8, gc_op=True)
        assert dev.branch_state.degenerate
        observable = qml.Hamiltonian([1.0, 0.5], [qml.Z(2), qml.Z(0)])
        return dev.exact_expval(observable).real

    def test_wire_sharing_swap_gradient_matches_finite_difference(self):
        theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        expectation = self._wire_sharing_expval_torch(theta)
        expectation.backward()
        grad_autograd = theta.grad

        assert torch.isfinite(grad_autograd)
        step = 1e-5
        with torch.no_grad():
            plus = self._wire_sharing_expval_torch(torch.tensor(0.5 + step, dtype=torch.float64))
            minus = self._wire_sharing_expval_torch(torch.tensor(0.5 - step, dtype=torch.float64))
        grad_finite_difference = (plus - minus) / (2 * step)
        torch.testing.assert_close(grad_autograd, grad_finite_difference, atol=ATOL_MATRIX_COMPARISON, rtol=0.0)

    def test_wire_sharing_swap_expval_gradcheck(self):
        theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(self._wire_sharing_expval_torch, (theta,), eps=1e-6, atol=1e-5)

    @staticmethod
    def _wire_sharing_prob_torch(theta):
        # Same degenerate circuit as _wire_sharing_expval_torch, read out through the full-outcome
        # probability path (reference relay + lifted amplitudes) instead of the expval path.
        n = 3
        dev = SwapAugmentedFermionicDevice(wires=n)

        def generator():
            yield qml.BasisState(np.array([0, 1, 1]), wires=range(n))
            yield qml.IsingXX(theta, wires=[0, 1])
            yield qml.IsingYY(0.7, wires=[1, 2])
            yield qml.SWAP(wires=[0, 1])
            yield qml.IsingXX(theta, wires=[1, 2])
            yield qml.SWAP(wires=[1, 2])
            yield qml.IsingYY(0.9, wires=[0, 1])
            yield qml.SWAP(wires=[0, 1])

        dev.execute_generator(generator(), reset=True, n_ops=8, gc_op=True)
        assert dev.branch_state.degenerate
        return dev.get_states_probability(np.array([0, 1, 1]))

    def test_wire_sharing_swap_probability_gradcheck(self):
        theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(self._wire_sharing_prob_torch, (theta,), eps=1e-6, atol=1e-5)

    def test_full_probabilities_sum_to_one(self):
        n = 3
        x = np.random.default_rng(9).uniform(-2, 2, size=3)

        @qml.qnode(SwapAugmentedFermionicDevice(wires=n))
        def got(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        probs = np.asarray(got(x))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=ATOL_MATRIX_COMPARISON)
        assert np.all(probs >= -ATOL_MATRIX_COMPARISON)

    @staticmethod
    def _spy_full_distribution(monkeypatch):
        """Wrap the device's full-distribution helper with a call counter and return the counter list."""
        import matchcake.devices.swap_augmented_fermionic_device as module

        calls = []
        original = module.basis_states_probabilities

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "basis_states_probabilities", spy)
        return calls

    def test_full_distribution_uses_fast_path_and_matches_default_qubit(self, monkeypatch):
        # The full outcome set on shared wires routes through basis_states_probabilities exactly once,
        # and the result matches default.qubit and the per-outcome loop.
        n = 3
        x = np.random.default_rng(7).uniform(-2, 2, size=3)
        calls = self._spy_full_distribution(monkeypatch)

        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def got(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._single_swap(x)
            return qml.probs(wires=range(n))

        probs = np.asarray(got(x))
        assert len(calls) == 1  # fast path taken
        assert not dev.branch_state.degenerate
        np.testing.assert_allclose(probs, np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)

        target = SwapAugmentedFermionicDevice.states_to_binary(np.arange(2**n), n)
        loop = np.stack(
            [np.asarray(dev.get_states_probability(target[i], qml.wires.Wires(range(n)))) for i in range(2**n)]
        )
        np.testing.assert_allclose(probs, loop, atol=ATOL_MATRIX_COMPARISON)

    def test_partial_outcome_set_skips_fast_path(self, monkeypatch):
        # A strict subset of outcomes stays on the per-outcome loop, not the tree.
        n = 3
        x = np.random.default_rng(7).uniform(-2, 2, size=3)
        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def run(x):
            self._single_swap(x)
            return qml.expval(qml.PauliZ(0))

        run(x)
        calls = self._spy_full_distribution(monkeypatch)
        target = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]])  # not the full big-endian {0,1}^3 set
        probs = np.asarray(dev.get_states_probability(target, qml.wires.Wires(range(n))))
        assert len(calls) == 0  # loop path
        loop = np.stack([np.asarray(dev.get_states_probability(row, qml.wires.Wires(range(n)))) for row in target])
        np.testing.assert_allclose(probs, loop, atol=ATOL_MATRIX_COMPARISON)

    def test_degenerate_full_distribution_skips_fast_path(self, monkeypatch):
        # A wire-sharing (degenerate) circuit must not use the tree fast path; it routes to the string engine.
        n = 3
        x = np.random.default_rng(0).uniform(-2, 2, size=5)
        calls = self._spy_full_distribution(monkeypatch)

        dev = SwapAugmentedFermionicDevice(wires=n)

        @qml.qnode(dev)
        def got(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        @qml.qnode(qml.device("default.qubit", wires=n))
        def ref(x):
            self._wire_sharing_swaps(x)
            return qml.probs(wires=range(n))

        probs = np.asarray(got(x))
        assert dev.branch_state.degenerate
        assert len(calls) == 0  # fast path bypassed for the degenerate state
        np.testing.assert_allclose(probs, np.asarray(ref(x)), atol=ATOL_MATRIX_COMPARISON)
