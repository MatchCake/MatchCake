import inspect

import numpy as np
import pennylane as qml
import pytest
import tqdm
from pennylane import BasisState
from pennylane.exceptions import DeviceError

from matchcake import NonInteractingFermionicDevice
from matchcake.observables.batch_hamiltonian import BatchHamiltonian
from matchcake.observables.batch_projector import BatchProjector
from matchcake.operations import CompRxRx, CompRyRy, CompZX, SptmCompRxRx, SptmCompRyRy
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompHH,
    SptmCompZX,
    SptmFermionicSuperposition,
)
from matchcake.operations.state_preparation.state_prep_from_gates import StatePrepFromGates

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)
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

    @pytest.mark.parametrize("parameter_name", ["pfaffian_chunk_size", "show_progress"])
    def test_parameter_is_explicit_keyword_only(self, parameter_name):
        parameters = inspect.signature(NonInteractingFermionicDevice.__init__).parameters
        assert parameter_name in parameters, f"{parameter_name} must be explicit, not hidden in **kwargs."
        assert parameters[parameter_name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[parameter_name].default is None

    @pytest.mark.parametrize("parameter_name, value", [("pfaffian_chunk_size", 4), ("show_progress", True)])
    def test_parameter_is_not_captured_by_init_kwargs(self, parameter_name, value):
        dev = NonInteractingFermionicDevice(wires=2, **{parameter_name: value})
        assert parameter_name not in dev._init_kwargs

    @pytest.mark.parametrize("chunk_size", [None, 1, 8])
    def test_pfaffian_chunk_size_is_stored_verbatim(self, chunk_size):
        dev = NonInteractingFermionicDevice(wires=2, pfaffian_chunk_size=chunk_size)
        assert dev.pfaffian_chunk_size == chunk_size

    def test_show_progress_defaults_to_false_without_p_bar(self):
        dev = NonInteractingFermionicDevice(wires=2)
        assert dev.show_progress is False

    def test_show_progress_defaults_to_true_with_p_bar(self):
        bar = tqdm.tqdm(total=5, disable=True)
        dev = NonInteractingFermionicDevice(wires=2, p_bar=bar)
        assert dev.show_progress is True
        bar.close()

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_show_progress_explicit_value_overrides_p_bar_default(self, show_progress):
        bar = tqdm.tqdm(total=5, disable=True)
        dev = NonInteractingFermionicDevice(wires=2, p_bar=bar, show_progress=show_progress)
        assert dev.show_progress is show_progress
        bar.close()

    @pytest.mark.parametrize("p_bar_given, expected", [(False, False), (True, True)])
    def test_explicit_none_show_progress_resolves_to_the_p_bar_default(self, p_bar_given, expected):
        bar = tqdm.tqdm(total=5, disable=True) if p_bar_given else None
        dev = NonInteractingFermionicDevice(wires=2, p_bar=bar, show_progress=None)
        assert dev.show_progress is expected
        if bar is not None:
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
        observable = qml.Projector(np.array([1, 0, 0, 1]) / np.sqrt(2), wires=[0, 1])
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
        np.testing.assert_allclose(dev._global_sptm.sptm_matrix(), np.eye(4))

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
        assert sptm.sptm_matrix().shape == (1, 4, 4)

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

    @pytest.mark.parametrize("n_wires", [2, 3])
    def test_convert_op_to_supported_adjoint_sptm(self, n_wires):
        dev = NonInteractingFermionicDevice(wires=n_wires)
        sptm = SingleParticleTransitionMatrixOperation.random(wires=range(n_wires), seed=TEST_SEED)
        converted = dev.convert_op_to_supported(qml.adjoint(sptm))
        assert isinstance(converted, SingleParticleTransitionMatrixOperation)
        np.testing.assert_allclose(
            converted.sptm_matrix(),
            sptm.adjoint().sptm_matrix(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_convert_op_to_supported_nested_adjoint_sptm(self):
        dev = NonInteractingFermionicDevice(wires=2)
        sptm = SptmCompRxRx(np.array([0.1, 0.4]), wires=[0, 1])
        converted = dev.convert_op_to_supported(qml.adjoint(qml.adjoint(sptm)))
        assert isinstance(converted, SingleParticleTransitionMatrixOperation)
        np.testing.assert_allclose(
            converted.sptm_matrix(),
            sptm.sptm_matrix(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_convert_op_to_supported_adjoint_native_matchgate(self):
        from matchcake.operations.matchgate_operation import MatchgateOperation

        dev = NonInteractingFermionicDevice(wires=2)
        converted = dev.convert_op_to_supported(qml.adjoint(qml.IsingXX(0.5, wires=[0, 1])))
        assert isinstance(converted, MatchgateOperation)

    def test_has_sptm_interface_to_sptm_operation_only(self):
        class _OpWithToSptm:
            def to_sptm_operation(self):
                return None

        assert NonInteractingFermionicDevice._has_sptm_interface(_OpWithToSptm()) is True

    def test_has_sptm_interface_sptm_attribute_only(self):
        class _OpWithSptmAttribute:
            single_particle_transition_matrix = None

        assert NonInteractingFermionicDevice._has_sptm_interface(_OpWithSptmAttribute()) is True

    def test_has_sptm_interface_plain_operator(self):
        assert NonInteractingFermionicDevice._has_sptm_interface(qml.IsingXX(0.5, wires=[0, 1])) is False

    @pytest.mark.parametrize("n_adjoints", [1, 2, 3])
    def test_qnode_adjoint_sptm_matches_unitary_adjoint(self, n_adjoints):
        params = np.array([0.1, 0.4])
        unitary = qml.math.to_numpy(qml.matrix(CompRxRx(params, wires=[0, 1])))
        if n_adjoints % 2 == 1:
            unitary = unitary.conj().T

        def apply_nested_adjoint():
            op = SptmCompRxRx(params, wires=[0, 1])
            for _ in range(n_adjoints):
                op = qml.adjoint(op)
            return op

        def make_circuit(device, apply_gate):
            @qml.qnode(device)
            def circuit():
                qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
                qml.IsingXY(0.4, wires=[0, 1])
                qml.IsingXY(0.9, wires=[1, 2])
                apply_gate()
                return qml.probs(wires=[0, 1, 2])

            return circuit

        expected = make_circuit(
            qml.device("default.qubit", wires=3),
            lambda: qml.QubitUnitary(unitary, wires=[0, 1]),
        )()
        actual = make_circuit(NonInteractingFermionicDevice(wires=3), apply_nested_adjoint)()
        np.testing.assert_allclose(
            qml.math.to_numpy(actual),
            qml.math.to_numpy(expected),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wrapper, expected_unitary",
        [
            ("pow2", lambda u, v: u @ u),
            ("pow3", lambda u, v: u @ u @ u),
            ("pow-1", lambda u, v: u.conj().T),
            ("pow-2", lambda u, v: (u @ u).conj().T),
            ("pow0", lambda u, v: np.eye(4)),
            ("prod_same", lambda u, v: u @ u),
            ("prod_mixed", lambda u, v: v @ u),
            ("prod_matchgate", lambda u, v: u @ v),
            ("sprod", lambda u, v: u),
            ("adjoint_pow", lambda u, v: (u @ u).conj().T),
            ("adjoint_prod", lambda u, v: (v @ u).conj().T),
            ("pow_eager-2", lambda u, v: (u @ u).conj().T),
            ("pow_eager3", lambda u, v: u @ u @ u),
            ("pow_int64", lambda u, v: u @ u),
            ("pow_float", lambda u, v: u @ u),
            ("pow_adjoint-2", lambda u, v: u @ u),
        ],
    )
    def test_qnode_wrapped_sptm_matches_unitary(self, wrapper, expected_unitary):
        rxrx_params = np.array([0.37, 1.1])
        ryry_params = np.array([0.2, -0.5])
        rxrx_unitary = qml.math.to_numpy(qml.matrix(CompRxRx(rxrx_params, wires=[0, 1])))
        ryry_unitary = qml.math.to_numpy(qml.matrix(CompRyRy(ryry_params, wires=[0, 1])))

        def rxrx():
            return SptmCompRxRx(rxrx_params, wires=[0, 1])

        def ryry():
            return SptmCompRyRy(ryry_params, wires=[0, 1])

        # Each entry builds its own operators so no unused operator gets queued in the QNode.
        wrapped = {
            "pow2": lambda: qml.pow(rxrx(), 2),
            "pow3": lambda: qml.pow(rxrx(), 3),
            "pow-1": lambda: qml.pow(rxrx(), -1),
            "pow-2": lambda: qml.pow(rxrx(), -2),
            "pow0": lambda: qml.pow(rxrx(), 0),
            "prod_same": lambda: qml.prod(rxrx(), rxrx()),
            "prod_mixed": lambda: qml.prod(ryry(), rxrx()),
            "prod_matchgate": lambda: qml.prod(rxrx(), CompRyRy(ryry_params, wires=[0, 1])),
            "sprod": lambda: qml.s_prod(1.0, rxrx()),
            "adjoint_pow": lambda: qml.adjoint(qml.pow(rxrx(), 2)),
            "adjoint_prod": lambda: qml.adjoint(qml.prod(ryry(), rxrx())),
            "pow_eager-2": lambda: qml.pow(rxrx(), -2, lazy=False),
            "pow_eager3": lambda: qml.pow(rxrx(), 3, lazy=False),
            "pow_int64": lambda: qml.pow(rxrx(), np.int64(2)),
            "pow_float": lambda: qml.pow(rxrx(), 2.0),
            "pow_adjoint-2": lambda: qml.pow(qml.adjoint(rxrx()), -2),
        }

        def make_circuit(device, apply_gate):
            @qml.qnode(device)
            def circuit():
                qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
                qml.IsingXY(0.4, wires=[0, 1])
                qml.IsingXY(0.9, wires=[1, 2])
                apply_gate()
                return qml.probs(wires=[0, 1, 2])

            return circuit

        expected = make_circuit(
            qml.device("default.qubit", wires=3),
            lambda: qml.QubitUnitary(expected_unitary(rxrx_unitary, ryry_unitary), wires=[0, 1]),
        )()
        actual = make_circuit(NonInteractingFermionicDevice(wires=3), wrapped[wrapper])()
        np.testing.assert_allclose(
            qml.math.to_numpy(actual),
            qml.math.to_numpy(expected),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_convert_op_to_supported_integer_pow_sptm(self):
        dev = NonInteractingFermionicDevice(wires=2)
        sptm = SptmCompRxRx(np.array([0.1, 0.4]), wires=[0, 1])
        converted = dev.convert_op_to_supported(qml.pow(sptm, 2))
        assert isinstance(converted, SingleParticleTransitionMatrixOperation)
        np.testing.assert_allclose(
            converted.sptm_matrix(),
            (sptm @ sptm).sptm_matrix(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_update_single_particle_transition_matrix_with_matchgate_operands(self):
        from matchcake.operations.matchgate_operation import MatchgateOperation

        old_op = MatchgateOperation.random(wires=[0, 1], seed=TEST_SEED)
        new_op = qml.adjoint(MatchgateOperation.random(wires=[0, 1], seed=TEST_SEED + 1))
        updated = NonInteractingFermionicDevice.update_single_particle_transition_matrix(old_op, new_op)
        old_sptm = SingleParticleTransitionMatrixOperation.from_operation(old_op).sptm_matrix()
        new_sptm = SingleParticleTransitionMatrixOperation.from_operation(new_op).sptm_matrix()
        np.testing.assert_allclose(
            qml.math.to_numpy(updated),
            qml.math.to_numpy(
                NonInteractingFermionicDevice.update_single_particle_transition_matrix(old_sptm, new_sptm)
            ),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "make_sptm",
        [
            lambda: SptmCompZX(wires=[0, 1]),
            lambda: SptmCompZX(wires=[1, 0]),
            lambda: SptmCompZX(wires=[0, 2]),
            lambda: SptmCompHH(wires=[1, 2]),
            lambda: SptmFermionicSuperposition(wires=[0, 1]),
            lambda: SptmFermionicSuperposition(wires=[0, 1, 2, 3]),
            lambda: SptmCompRxRx(np.array([np.pi, np.pi]), wires=[0, 1]),
            lambda: SingleParticleTransitionMatrixOperation(-np.eye(8), wires=[0, 1, 2, 3]),
            lambda: SingleParticleTransitionMatrixOperation(np.diag([-1.0, 1, 1, 1, 1, -1, 1, 1]), wires=[0, 1, 2, 3]),
            lambda: SingleParticleTransitionMatrixOperation.random(wires=[0, 1, 2, 3], seed=TEST_SEED),
        ],
    )
    def test_default_qubit_decomposition_matches_nif(self, make_sptm):
        n_wires = 4

        def make_circuit(device):
            @qml.qnode(device)
            def circuit():
                qml.BasisState(np.array([1, 0, 1, 0]), wires=range(n_wires))
                for wire in range(n_wires - 1):
                    qml.IsingXY(0.3 + 0.2 * wire, wires=[wire, wire + 1])
                qml.apply(make_sptm())
                for wire in range(n_wires - 1):
                    qml.IsingXY(0.7 - 0.1 * wire, wires=[wire, wire + 1])
                return qml.probs(wires=range(n_wires))

            return circuit

        np.testing.assert_allclose(
            qml.math.to_numpy(make_circuit(qml.device("default.qubit", wires=n_wires))()),
            qml.math.to_numpy(make_circuit(NonInteractingFermionicDevice(wires=n_wires))()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_convert_op_to_supported_fractional_pow_sptm_raises(self):
        dev = NonInteractingFermionicDevice(wires=2)
        with pytest.raises(DeviceError):
            dev.convert_op_to_supported(qml.pow(SptmCompRxRx(np.array([0.1, 0.4]), wires=[0, 1]), 0.5))

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
