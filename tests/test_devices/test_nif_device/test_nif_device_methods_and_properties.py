import numpy as np
import pennylane as qml
import pytest
import tqdm
from pennylane import BasisState
from pennylane.exceptions import DeviceError
from pennylane.ops.op_math import Sum
from pennylane.tape import QuantumTape

from matchcake import NonInteractingFermionicDevice
from matchcake.observables.batch_hamiltonian import BatchHamiltonian
from matchcake.observables.batch_projector import BatchProjector
from matchcake.operations import CompZX
from matchcake.operations.state_preparation import ProductState
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

    def test_global_sptm_batched_returns_3d_matrix(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev._batched = True
        dev._global_sptm = None
        sptm = dev.global_sptm
        assert sptm.matrix().shape == (1, 4, 4)

    def test_patched_super_batch_transform_use_grouping_false(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.use_grouping = False
        sum_obs = qml.sum(qml.Z(0), qml.Z(1))
        with QuantumTape() as tape:
            qml.BasisState([0, 0], wires=[0, 1])
            qml.expval(sum_obs)
        circuits, fn = dev._patched_super_batch_transform(tape)
        assert len(circuits) == 1

    def test_patched_super_batch_transform_grouping_known(self):
        dev = NonInteractingFermionicDevice(wires=2)
        ham = qml.Hamiltonian([1.0, 0.5], [qml.Z(0) @ qml.Z(1), qml.Z(0)])
        ham.compute_grouping()
        with QuantumTape() as tape:
            qml.BasisState([0, 0], wires=[0, 1])
            qml.expval(ham)
        circuits, fn = dev._patched_super_batch_transform(tape)
        assert len(circuits) >= 1

    def test_patched_super_batch_transform_hamiltonian_wires(self):
        dev = NonInteractingFermionicDevice(wires=2)
        ham = qml.Hamiltonian([1.0, 0.5], [qml.Z(0), qml.Z(1)])
        with QuantumTape() as tape:
            qml.BasisState([0, 0], wires=[0, 1])
            qml.expval(ham)
        circuits, fn = dev._patched_super_batch_transform(tape)
        assert len(circuits) >= 1

    def test_batch_transform_batch_hamiltonian_none_batch_size(self):
        dev = NonInteractingFermionicDevice(wires=2)
        batch_ham = BatchHamiltonian([1.0, 0.5], [qml.Z(0), qml.Z(1)])
        with QuantumTape() as tape:
            qml.BasisState([0, 0], wires=[0, 1])
            qml.expval(batch_ham)
        circuits, fn = dev.batch_transform(tape)
        assert len(circuits) == 1
