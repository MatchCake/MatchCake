import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError

from matchcake import NonInteractingFermionicDevice, SwapAugmentedFermionicDevice
from matchcake.operations.state_preparation import ProductState

from ...configs import ATOL_MATRIX_COMPARISON


class TestSwapAugmentedFermionicDevice:
    """End-to-end device tests through the ``qml`` API for the verified regime.

    Covered: zero-SWAP (strict superset of ``nif.qubit``), a single SWAP, and SWAPs on disjoint
    wire pairs, for basis and product-state inputs, ``probs``/``expval``/marginals/batched params.
    Wire-sharing SWAPs are a known open case (see ``swap_injection_math.md`` section 12) and are not
    asserted here.
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
