import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import Rxx

from ...configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED, set_seed

_DTYPE_CASES = [
    pytest.param(torch.float64, "float64", "complex128", id="float64"),
    pytest.param(torch.float32, "float32", "complex64", id="float32"),
]


class TestNIFDeviceDtype:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @staticmethod
    def _probs_circuit(device: NonInteractingFermionicDevice):
        @qml.qnode(device)
        def circuit(x):
            Rxx(x, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        return circuit

    @staticmethod
    def _expval_circuit(device: NonInteractingFermionicDevice):
        @qml.qnode(device)
        def circuit(x):
            Rxx(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        return circuit

    def test_default_dtypes_are_complex128_and_float64(self):
        device = NonInteractingFermionicDevice(wires=2)
        assert device.C_DTYPE == torch.complex128
        assert device.R_DTYPE == torch.float64

    def test_instance_dtype_does_not_mutate_class_default(self):
        NonInteractingFermionicDevice(wires=2, c_dtype=torch.complex64, r_dtype=torch.float32)
        assert NonInteractingFermionicDevice.C_DTYPE == torch.complex128
        assert NonInteractingFermionicDevice.R_DTYPE == torch.float64

    def test_r_dtype_and_c_dtype_are_independent(self):
        device = NonInteractingFermionicDevice(wires=2, c_dtype=torch.complex64, r_dtype=torch.float64)
        assert device.C_DTYPE == torch.complex64
        assert device.R_DTYPE == torch.float64

    def test_numpy_dtype_argument_is_normalized_to_torch(self):
        device = NonInteractingFermionicDevice(wires=2, c_dtype=np.complex64, r_dtype=np.float32)
        assert device.C_DTYPE == torch.complex64
        assert device.R_DTYPE == torch.float32

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_global_sptm_is_real_and_transition_matrix_is_complex(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        self._probs_circuit(device)(torch.tensor([0.1, 0.2], dtype=torch.float64))
        assert device.global_sptm.dtype == real_name
        assert qml.math.get_dtype_name(device.transition_matrix) == complex_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_global_sptm_identity_fallback_is_real(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        assert device.global_sptm.dtype == real_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_global_sptm_setter_normalizes_to_real_r_dtype(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        device.global_sptm = np.eye(4, dtype=np.complex128)
        assert device.global_sptm.dtype == real_name
        assert qml.math.get_dtype_name(device.transition_matrix) == complex_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_global_sptm_stays_real_after_each_applied_operation(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=3, r_dtype=r_dtype)
        angles = torch.tensor([0.1, 0.2], dtype=torch.float64)
        operations = [Rxx(angles, wires=[0, 1]), Rxx(angles, wires=[1, 2]), Rxx(angles, wires=[0, 1])]
        for operation in operations:
            device.apply_op(operation)
            assert device.global_sptm.dtype == real_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_global_sptm_real_through_apply_generator(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=3, r_dtype=r_dtype)
        angles = torch.tensor([0.1, 0.2], dtype=torch.float64)
        device.apply_generator([Rxx(angles, wires=[0, 1]), Rxx(angles, wires=[1, 2]), Rxx(angles, wires=[0, 1])])
        assert device.global_sptm.dtype == real_name
        assert qml.math.get_dtype_name(device.transition_matrix) == complex_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_probs_dtype_matches_r_dtype_precision(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        probs = self._probs_circuit(device)(torch.tensor([0.1, 0.2], dtype=torch.float64))
        assert qml.math.get_dtype_name(probs) == real_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_expval_is_real(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        expval = self._expval_circuit(device)(torch.tensor([0.1, 0.2], dtype=torch.float64))
        assert "complex" not in qml.math.get_dtype_name(expval)

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_qnode_probs_preserve_precision_with_matching_input(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        probs = self._probs_circuit(device)(torch.tensor([0.1, 0.2], dtype=r_dtype))
        assert qml.math.get_dtype_name(probs) == real_name

    @pytest.mark.parametrize("r_dtype, real_name, complex_name", _DTYPE_CASES)
    def test_qnode_gradient_preserves_precision(self, r_dtype, real_name, complex_name):
        device = NonInteractingFermionicDevice(wires=2, r_dtype=r_dtype)
        circuit = self._probs_circuit(device)
        angles = torch.tensor([0.1, 0.2], dtype=r_dtype, requires_grad=True)
        circuit(angles).sum().backward()
        assert qml.math.get_dtype_name(angles.grad) == real_name

    def test_lower_precision_probs_match_default_within_tolerance(self):
        x = torch.tensor([0.1, 0.2], dtype=torch.float64)
        probs_default = self._probs_circuit(NonInteractingFermionicDevice(wires=2))(x)
        probs_low = self._probs_circuit(NonInteractingFermionicDevice(wires=2, r_dtype=torch.float32))(x)
        np.testing.assert_allclose(
            probs_low.detach().numpy(),
            probs_default.detach().numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
