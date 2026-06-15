import numpy as np
import pennylane as qml
import pytest
import tqdm
from pennylane import BasisState
from pennylane.exceptions import DeviceError

from matchcake import NonInteractingFermionicDevice
from matchcake.observables.batch_hamiltonian import BatchHamiltonian
from matchcake.observables.batch_projector import BatchProjector
from matchcake.operations import CompZX
from matchcake.operations.state_preparation.state_prep_from_gates import StatePrepFromGates

from ...configs import TEST_SEED, set_seed
from .. import init_nif_device


def _make_zero_state_device(n_wires=2, shots=None):
    dev = init_nif_device(wires=n_wires, shots=shots)
    dev.apply_state_prep(BasisState(np.zeros(n_wires, dtype=int), wires=range(n_wires)))
    return dev


class TestNIFDeviceMethodsAndProperties:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_execute_generator_apply_auto_with_existing_sptm(self):
        dev = init_nif_device(wires=2)
        dev.apply_generator([CompZX(wires=[0, 1])])
        dev.apply_state_prep(BasisState(np.zeros(2, dtype=int), wires=[0, 1]))
        result = dev.execute_generator([CompZX(wires=[0, 1])], observable=None, output_type=None, apply="auto")
        assert result is None

    def test_execute_output_star_state(self):
        dev = _make_zero_state_device(n_wires=2, shots=16)
        dev.apply_generator([CompZX(wires=[0, 1])])
        result = dev.execute_output(output_type="star_state")
        assert result is not None

    def test_execute_output_star_state_alias(self):
        dev = _make_zero_state_device(n_wires=2, shots=16)
        dev.apply_generator([CompZX(wires=[0, 1])])
        result = dev.execute_output(output_type="*state")
        assert result is not None

    def test_execute_output_unsupported_type_raises(self):
        dev = _make_zero_state_device(n_wires=2)
        with pytest.raises(ValueError, match="not supported"):
            dev.execute_output(output_type="unsupported_type")

    def test_execute_output_samples_no_shots_raises(self):
        dev = _make_zero_state_device(n_wires=2, shots=None)
        with pytest.raises(ValueError, match="number of shots"):
            dev.execute_output(output_type="samples")

    def test_get_state_probability_int_input(self):
        dev = _make_zero_state_device(n_wires=2)
        prob = dev.get_state_probability(0, wires=[0, 1])
        np.testing.assert_allclose(prob, 1.0, atol=1e-6)

    def test_get_state_probability_list_input(self):
        dev = _make_zero_state_device(n_wires=2)
        prob = dev.get_state_probability([0, 0], wires=[0, 1])
        np.testing.assert_allclose(prob, 1.0, atol=1e-6)

    def test_get_state_probability_string_input(self):
        dev = _make_zero_state_device(n_wires=2)
        prob = dev.get_state_probability("00", wires=[0, 1])
        np.testing.assert_allclose(prob, 1.0, atol=1e-6)

    def test_get_states_probability_int_input(self):
        dev = _make_zero_state_device(n_wires=2)
        prob = dev.get_states_probability(0)
        assert prob is not None

    def test_get_states_probability_list_input(self):
        dev = _make_zero_state_device(n_wires=2)
        prob = dev.get_states_probability([0, 0])
        assert prob is not None

    def test_get_states_probability_string_input(self):
        dev = _make_zero_state_device(n_wires=2)
        prob = dev.get_states_probability("00")
        assert prob is not None

    def test_analytic_probability_3d_wires_raises(self):
        dev = _make_zero_state_device(n_wires=2)
        wires_3d = np.zeros((2, 2, 2), dtype=int)
        with pytest.raises(ValueError, match="1D or 2D"):
            dev.analytic_probability(wires=wires_3d)

    def test_analytic_probability_unbatched_normalized(self):
        dev = _make_zero_state_device(n_wires=2)
        probs = np.asarray(dev.analytic_probability(wires=[0, 1]))
        assert probs.shape == (4,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_analytic_probability_batched_normalized_per_row(self):
        from matchcake.operations import SptmCompRxRx

        n_wires, batch_size = 2, 3
        rng = np.random.default_rng(TEST_SEED)
        dev = init_nif_device(wires=n_wires)
        dev.apply_state_prep(BasisState(np.zeros(n_wires, dtype=int), wires=range(n_wires)))
        dev.apply([SptmCompRxRx(rng.uniform(0, np.pi, (batch_size, 2)), wires=[0, 1])])
        probs = np.asarray(dev.analytic_probability(wires=[0, 1]))
        assert probs.shape == (batch_size, 2**n_wires)
        # Each batch element must be a proper distribution (regression: previously a global
        # sum normalized every row to 1 / batch_size).
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(batch_size), atol=1e-6)

    def test_analytic_probability_batched_product_state_normalized_per_row(self):
        from matchcake.operations.state_preparation.product_state import ProductState

        n_wires, batch_size = 2, 3
        rng = np.random.default_rng(TEST_SEED)
        angles = rng.uniform(0.2, np.pi - 0.2, (batch_size, n_wires))
        amplitudes = np.stack([np.cos(angles / 2), np.sin(angles / 2)], axis=-1).astype(complex)
        dev = init_nif_device(wires=n_wires)
        dev.apply_state_prep(ProductState(amplitudes, wires=range(n_wires)))
        probs = np.asarray(dev.analytic_probability(wires=[0, 1]))
        assert probs.shape == (batch_size, 2**n_wires)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(batch_size), atol=1e-6)

    def test_exact_expval_batch_projector(self):
        dev = _make_zero_state_device(n_wires=2)
        states = [np.array([0, 0])]
        wires = [0, 1]
        projector = BatchProjector(states, wires=wires)
        result = dev.exact_expval(projector)
        assert result is not None

    def test_exact_expval_batch_hamiltonian(self):
        dev = _make_zero_state_device(n_wires=2)
        dev.apply_generator([CompZX(wires=[0, 1])])
        hamiltonian = BatchHamiltonian([1.0, 0.5], [qml.Z(0), qml.Z(1)])
        result = dev.exact_expval(hamiltonian)
        assert result is not None

    def test_expval_with_shots(self):
        dev = _make_zero_state_device(n_wires=2, shots=100)
        dev.apply_generator([CompZX(wires=[0, 1])])
        dev._samples = dev.generate_samples()
        obs = qml.Z(0) @ qml.Z(1)
        result = dev.expval(obs)
        assert result is not None

    def test_asarray_complex_input(self):
        dev = NonInteractingFermionicDevice(wires=2)
        x = np.array([1.0 + 0.5j, 0.0 + 1.0j])
        result = dev._asarray(x)
        assert result is not None

    def test_update_p_bar_with_none_p_bar(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.p_bar = None
        dev.update_p_bar(1)

    def test_p_bar_set_n_with_none(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.p_bar = None
        dev.p_bar_set_n(5)

    def test_p_bar_set_total_with_none(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.p_bar = None
        dev.p_bar_set_total(10)

    def test_p_bar_set_postfix_with_none(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.p_bar = None
        dev.p_bar_set_postfix({"key": "val"})

    def test_p_bar_set_postfix_str_with_none(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.p_bar = None
        dev.p_bar_set_postfix_str("msg")

    def test_close_p_bar_with_none(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.p_bar = None
        dev.close_p_bar()

    def test_initialize_p_bar_with_show_progress(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        bar = dev.initialize_p_bar(total=5, desc="test")
        assert isinstance(bar, tqdm.tqdm)
        bar.close()

    def test_update_p_bar_with_active_bar(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        dev.initialize_p_bar(total=5, desc="test")
        dev.update_p_bar(1)
        dev.close_p_bar()

    def test_p_bar_set_n_with_active_bar(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        dev.initialize_p_bar(total=5, desc="test")
        dev.p_bar_set_n(3)
        dev.close_p_bar()

    def test_p_bar_set_total_with_active_bar(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        dev.initialize_p_bar(total=5, desc="test")
        dev.p_bar_set_total(10)
        dev.close_p_bar()

    def test_p_bar_set_postfix_with_active_bar(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        dev.initialize_p_bar(total=5, desc="test")
        dev.p_bar_set_postfix({"step": 1})
        dev.close_p_bar()

    def test_p_bar_set_postfix_str_with_active_bar(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        dev.initialize_p_bar(total=5, desc="test")
        dev.p_bar_set_postfix_str("running")
        dev.close_p_bar()

    def test_close_p_bar_with_active_bar(self):
        dev = NonInteractingFermionicDevice(wires=2, show_progress=True)
        dev.initialize_p_bar(total=5, desc="test")
        dev.close_p_bar()
        assert dev.p_bar is not None

    def test_state_prep_op_property(self):
        dev = NonInteractingFermionicDevice(wires=2)
        assert dev.state_prep_op is not None

    def test_covariance_matrix_property(self):
        dev = _make_zero_state_device(n_wires=2)
        cov = dev.covariance_matrix
        assert cov.shape == (4, 4)

    def test_extended_covariance_matrix_property(self):
        dev = _make_zero_state_device(n_wires=2)
        ext_cov = dev.extended_covariance_matrix
        assert ext_cov.shape == (5, 5)

    def test_execute_generator_with_apply_auto_no_sptm(self):
        dev = init_nif_device(wires=2)
        dev.apply_state_prep(BasisState(np.zeros(2, dtype=int), wires=[0, 1]))
        result = dev.execute_generator([CompZX(wires=[0, 1])], observable=None, output_type="probs", apply="auto")
        assert result is not None

    def test_covariance_matrix_non_product_state_raises(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev._state_prep_op = BasisState(np.zeros(2, dtype=int), wires=[0, 1])
        with pytest.raises(ValueError, match="Covariance matrix"):
            _ = dev.covariance_matrix

    def test_extended_covariance_matrix_basis_state(self):
        dev = _make_zero_state_device(n_wires=2)
        dev._state_prep_op = BasisState(np.zeros(2, dtype=int), wires=[0, 1])
        ext_cov = dev.extended_covariance_matrix
        assert ext_cov.shape == (5, 5)

    def test_extended_covariance_matrix_invalid_type_raises(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev._state_prep_op = qml.Identity(wires=[0, 1])
        with pytest.raises(ValueError, match="Extended covariance matrix requires"):
            _ = dev.extended_covariance_matrix

    def test_compute_star_state_cached(self):
        dev = _make_zero_state_device(n_wires=2, shots=16)
        dev.apply_generator([CompZX(wires=[0, 1])])
        dev._samples = dev.generate_samples()
        r1 = dev.compute_star_state()
        r2 = dev.compute_star_state()
        assert r1 == r2

    def test_exact_expval_unhandled_observable_raises(self):
        dev = _make_zero_state_device(n_wires=2)
        state_prep = StatePrepFromGates(lambda wires: [qml.Hadamard(wires=wires[0])], wires=[0, 1])
        dev._state_prep_op = state_prep
        observable = qml.X(0) @ qml.Z(1)
        with pytest.raises(DeviceError):
            dev.exact_expval(observable)

    def test_apply_state_prep_non_basis_state_prep_base(self):
        dev = NonInteractingFermionicDevice(wires=2)
        from matchcake.operations.state_preparation import ProductState

        prod = ProductState.from_basis_state(np.array([0, 1]), wires=[0, 1])
        result = dev.apply_state_prep(prod, index=0)
        assert result is True
        assert isinstance(dev._state_prep_op, ProductState)

    def test_analytic_probability_default_wires(self):
        dev = _make_zero_state_device(n_wires=2)
        probs = dev.analytic_probability()
        assert probs.shape == (4,)
        np.testing.assert_allclose(float(probs[0]), 1.0, atol=1e-6)

    def test_sample_basis_states_1d_probs(self):
        dev = NonInteractingFermionicDevice(wires=2, shots=32)
        dev._current_shots = 32
        probs_1d = np.array([0.5, 0.5])
        samples = dev.sample_basis_states(2, probs_1d)
        assert samples.shape == (32,)

    def test_global_sptm_setter_with_operation(self):
        from matchcake.operations import SingleParticleTransitionMatrixOperation

        dev = NonInteractingFermionicDevice(wires=2)
        op = SingleParticleTransitionMatrixOperation(np.eye(4), wires=[0, 1])
        dev.global_sptm = op
        assert isinstance(dev._global_sptm, SingleParticleTransitionMatrixOperation)

    def test_global_sptm_setter_with_sptm_instance(self):
        from matchcake.operations import SingleParticleTransitionMatrixOperation

        dev = NonInteractingFermionicDevice(wires=2)
        sptm = SingleParticleTransitionMatrixOperation(np.eye(4), wires=[0, 1])
        dev.global_sptm = sptm
        assert dev._global_sptm.dtype == "float64"
        np.testing.assert_allclose(dev._global_sptm.matrix(), np.eye(4))

    def test_execute_single_quantum_script_directly(self):
        dev = NonInteractingFermionicDevice(wires=2)
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 0]), wires=[0, 1])],
            [qml.expval(qml.PauliZ(0))],
        )
        result = dev.execute(tape)
        np.testing.assert_allclose(float(result), 1.0, atol=1e-6)

    def test_probability_with_shots_via_qnode(self):
        dev = NonInteractingFermionicDevice(wires=2, shots=512)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()
        assert probs.shape == (4,)
        np.testing.assert_allclose(probs[0], 1.0, atol=0.05)

    def test_sample_with_observable_via_qnode(self):
        dev = NonInteractingFermionicDevice(wires=2, shots=64)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            return qml.sample(qml.PauliZ(0))

        samples = circuit()
        assert samples.shape == (64,)
        np.testing.assert_array_equal(samples, np.ones(64))

    def test_sample_basis_states_raises_without_shots(self):
        dev = NonInteractingFermionicDevice(wires=2)
        with pytest.raises(ValueError, match="shots"):
            dev.sample_basis_states(2, np.array([0.5, 0.5]))

    def test_sample_basis_states_batched_2d_probs(self):
        dev = NonInteractingFermionicDevice(wires=2, shots=16)
        dev._current_shots = 16
        probs_2d = np.array([[0.5, 0.5], [0.25, 0.75]])
        samples = dev.sample_basis_states(2, probs_2d)
        assert samples.shape == (2, 16)

    def test_setup_execution_config_default(self):
        dev = NonInteractingFermionicDevice(wires=2)
        config = dev.setup_execution_config()
        assert config.gradient_method == "backprop"

    def test_supports_derivatives_no_config(self):
        dev = NonInteractingFermionicDevice(wires=2)
        assert dev.supports_derivatives() is True

    def test_supports_derivatives_unsupported_method(self):
        from pennylane.devices import ExecutionConfig

        dev = NonInteractingFermionicDevice(wires=2)
        config = ExecutionConfig(gradient_method="adjoint")
        assert dev.supports_derivatives(config) is False

    def test_asarray_real_input_with_explicit_dtype(self):
        dev = NonInteractingFermionicDevice(wires=2)
        x = np.array([1.0, 2.0, 3.0])
        result = dev._asarray(x, dtype=dev.R_DTYPE)
        assert result is not None

    def test_global_sptm_batched_returns_3d_matrix(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev._batched = True
        dev._global_sptm = None
        sptm = dev.global_sptm
        assert sptm.matrix().shape == (1, 4, 4)

    def test_convert_op_to_supported_passthrough_matchgate(self):
        from matchcake.operations.matchgate_operation import MatchgateOperation

        dev = NonInteractingFermionicDevice(wires=2)
        op = MatchgateOperation.random(wires=[0, 1], seed=TEST_SEED)
        assert dev.convert_op_to_supported(op) is op

    def test_convert_op_to_supported_passthrough_sptm(self):
        from matchcake.operations import SingleParticleTransitionMatrixOperation

        dev = NonInteractingFermionicDevice(wires=2)
        op = SingleParticleTransitionMatrixOperation(np.eye(4), wires=[0, 1])
        assert dev.convert_op_to_supported(op) is op

    def test_convert_op_to_supported_converts_native_matchgate(self):
        from matchcake.operations.matchgate_operation import MatchgateOperation

        dev = NonInteractingFermionicDevice(wires=2)
        converted = dev.convert_op_to_supported(qml.IsingXX(0.5, wires=[0, 1]))
        assert isinstance(converted, MatchgateOperation)

    def test_convert_op_to_supported_non_matchgate_raises(self):
        dev = NonInteractingFermionicDevice(wires=2)
        with pytest.raises(DeviceError, match="MatchgateOperation or SingleParticleTransitionMatrixOperation"):
            dev.convert_op_to_supported(qml.CNOT(wires=[0, 1]))

    def test_apply_generator_converts_native_matchgate(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.apply_generator([qml.IsingXX(0.5, wires=[0, 1])])
        assert dev.global_sptm is not None

    def test_stopping_condition_native_matchgate(self):
        dev = NonInteractingFermionicDevice(wires=2)
        assert dev._stopping_condition(qml.IsingXX(0.5, wires=[0, 1])) is True
        assert dev._stopping_condition(qml.CNOT(wires=[0, 1])) is False

    def test_qnode_with_native_matchgate(self):
        dev = NonInteractingFermionicDevice(wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            qml.IsingXX(theta, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        reference = qml.device("default.qubit", wires=2)

        @qml.qnode(reference)
        def reference_circuit(theta):
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            qml.IsingXX(theta, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        np.testing.assert_allclose(np.asarray(circuit(0.5)), np.asarray(reference_circuit(0.5)), atol=1e-6)

    def test_qnode_with_non_matchgate_raises(self):
        dev = NonInteractingFermionicDevice(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([0, 0]), wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        with pytest.raises(DeviceError):
            circuit()

    def test_exact_expval_falls_back_to_probabilities(self):
        # Z-only weight-3 string with a non-(Basis/Product) state prep: the M-Pfaffian and
        # Clifford strategies decline, so exact_expval falls back to the probability strategy.
        dev = NonInteractingFermionicDevice(wires=3)
        state_prep = StatePrepFromGates(lambda wires: [qml.PauliX(wires=wires[0])], wires=[0, 1, 2])
        dev._state_prep_op = state_prep
        observable = qml.Z(0) @ qml.Z(1) @ qml.Z(2)
        result = dev.exact_expval(observable)
        np.testing.assert_allclose(np.real(np.asarray(result)).squeeze(), -1.0)
