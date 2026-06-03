import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.operations import MatchgateOperation, Rxx
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRyRy,
)
from matchcake.utils import torch_utils
from matchcake.utils.math import svd

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


@pytest.mark.parametrize(
    "batch_size, size",
    [(batch_size, size) for batch_size in [1, 3] for size in [2, 3, 4]],
)
class TestSingleParticleTransitionMatrixOperation:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.fixture
    def input_matrix(self, batch_size, size):
        return np.random.random((batch_size, 2 * size, 2 * size))

    @pytest.mark.parametrize(
        "matrix_dtype, expected_name",
        [
            (np.complex128, "float64"),
            (np.complex64, "float32"),
            (np.float64, "float64"),
            (np.float32, "float32"),
            (torch.complex64, "float32"),
            (torch.complex128, "float64"),
        ],
    )
    def test_dtype_property_returns_real_backend_agnostic_name(self, batch_size, size, matrix_dtype, expected_name):
        # The SPTM is real-valued: a complex input is stored as its real counterpart, so the dtype
        # property always reports a floating-point dtype name.
        shape = (batch_size, 2 * size, 2 * size)
        if isinstance(matrix_dtype, torch.dtype):
            matrix = torch.zeros(shape, dtype=matrix_dtype)
        else:
            matrix = np.zeros(shape, dtype=matrix_dtype)
        operation = SingleParticleTransitionMatrixOperation(matrix, wires=np.arange(size))
        assert operation.dtype == expected_name

    @pytest.mark.parametrize("matrix_dtype, expected_name", [(np.float32, "float32"), (np.float64, "float64")])
    def test_to_torch_preserves_matrix_dtype(self, batch_size, size, matrix_dtype, expected_name):
        matrix = np.random.random((batch_size, 2 * size, 2 * size)).astype(matrix_dtype)
        operation = SingleParticleTransitionMatrixOperation(matrix, wires=np.arange(size)).to_torch()
        assert isinstance(operation.matrix(), torch.Tensor)
        assert operation.dtype == expected_name

    def test_matrix_is_always_real(self, batch_size, size):
        complex_matrix = np.random.random((batch_size, 2 * size, 2 * size)).astype(np.complex128)
        operation = SingleParticleTransitionMatrixOperation(complex_matrix, wires=np.arange(size))
        assert "complex" not in operation.dtype
        assert "complex" not in qml.math.get_dtype_name(operation.matrix())

    def test_constructor_drops_negligible_imaginary_part(self, batch_size, size):
        real_part = np.random.random((batch_size, 2 * size, 2 * size))
        complex_matrix = real_part + 1e-9j * np.random.random((batch_size, 2 * size, 2 * size))
        operation = SingleParticleTransitionMatrixOperation(complex_matrix, wires=np.arange(size))
        np.testing.assert_allclose(
            qml.math.cast(operation.matrix(), "float64"),
            real_part,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_constructor_raises_on_significant_imaginary_part(self, batch_size, size):
        complex_matrix = np.random.random((batch_size, 2 * size, 2 * size)).astype(np.complex128)
        complex_matrix += 1j * np.random.random((batch_size, 2 * size, 2 * size))
        with pytest.raises(ValueError, match="must be real"):
            SingleParticleTransitionMatrixOperation(complex_matrix, wires=np.arange(size), check_real=True)

    def test_constructor_does_not_check_imaginary_when_check_real_false(self, batch_size, size):
        complex_matrix = np.random.random((batch_size, 2 * size, 2 * size)).astype(np.complex128)
        complex_matrix += 1j * np.random.random((batch_size, 2 * size, 2 * size))
        operation = SingleParticleTransitionMatrixOperation(complex_matrix, wires=np.arange(size), check_real=False)
        assert "complex" not in operation.dtype

    def test_gradient_flows_through_complex_input(self, batch_size, size):
        def sptm_from_complex(real_matrix):
            complex_matrix = qml.math.cast(real_matrix, "complex128")
            return SingleParticleTransitionMatrixOperation(complex_matrix, wires=np.arange(size)).matrix()

        assert torch.autograd.gradcheck(
            sptm_from_complex,
            torch_utils.to_tensor(np.random.random((batch_size, 2 * size, 2 * size)), torch.double).requires_grad_(),
            raise_exception=True,
            check_undefined_grad=False,
        )

    def test_gradient_flows_through_matchgate_built_sptm(self, batch_size, size):
        # Building an SPTM from a matchgate goes through a complex matrix whose imaginary part is
        # numerically zero. The gradient must still flow back to the (real) matchgate angles.
        def sptm_from_matchgate(angles):
            return SingleParticleTransitionMatrixOperation.from_operation(Rxx(angles, wires=[0, 1])).matrix()

        assert torch.autograd.gradcheck(
            sptm_from_matchgate,
            torch_utils.to_tensor(np.random.random(batch_size), torch.double).requires_grad_(),
            raise_exception=True,
            check_undefined_grad=False,
        )

    def test_sptm_sum_gradient_check(self, input_matrix):
        def sptm_sum(p):
            return torch.sum(
                SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2)).matrix()
            )

        assert torch.autograd.gradcheck(
            sptm_sum,
            torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
            raise_exception=True,
            check_undefined_grad=False,
        )

    def test_sptm_init_gradient_check(self, input_matrix):
        def sptm_init(p):
            return SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2)).matrix()

        assert torch.autograd.gradcheck(
            sptm_init,
            torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
            raise_exception=True,
            check_undefined_grad=False,
        )

    def test_sptm_copy_gradient_check(self, input_matrix):
        def func(p):
            wires = np.arange(0, p.shape[-1] // 2, dtype=int)
            op = SingleParticleTransitionMatrixOperation(matrix=p, wires=wires)
            new_op = SingleParticleTransitionMatrixOperation(matrix=op.matrix(), wires=op.wires)
            return new_op.matrix()

        assert torch.autograd.gradcheck(
            func,
            torch_utils.to_tensor(input_matrix, torch.double).requires_grad_(),
            raise_exception=True,
            check_undefined_grad=False,
        )

    def test_to_matchgate(self, batch_size, size):
        n_qubits = 2
        mgo = MatchgateOperation.random(batch_size=batch_size, wires=np.arange(n_qubits), seed=size)
        sptm = SingleParticleTransitionMatrixOperation.from_operation(mgo)
        pred_mgo = sptm.to_matchgate()
        phased_identity = torch.einsum("...ij,...jk->ik", torch.linalg.inv(pred_mgo.matrix()), mgo.matrix())
        diag = torch.diag(phased_identity)
        diag_diff = torch.diff(diag)
        torch.testing.assert_close(diag_diff, torch.zeros_like(diag_diff), msg=f"diag is not a scalar: {diag}")
        phase = -1j * torch.log(phased_identity[..., 0, 0])
        # torch.testing.assert_close(phase.imag, torch.zeros_like(phase.imag), msg=f"phase is not real: {phase}")
        target_phased_identity = torch.zeros_like(phased_identity)
        target_phased_identity[..., np.arange(2 * n_qubits), np.arange(2 * n_qubits)] = torch.exp(1j * phase)
        torch.testing.assert_close(phased_identity, target_phased_identity, msg="exp(i * p) != U^(-1) V")

    def test_to_qubit_unitary(self, batch_size, size):
        sptm = SingleParticleTransitionMatrixOperation.random(batch_size=batch_size, wires=np.arange(size), seed=size)
        unitary = sptm.to_qubit_unitary()
        assert unitary._unitary_check(unitary.matrix(), int(2 ** len(unitary.wires)))

    def test_compute_decomposition(self, batch_size, size):
        sptm = SingleParticleTransitionMatrixOperation.random(batch_size=batch_size, wires=np.arange(size), seed=size)
        [decomposition] = sptm.decomposition()
        torch.testing.assert_close(
            sptm.to_qubit_unitary().matrix(),
            decomposition.matrix(),
            msg="Reconstructed SPTM does not match the original",
        )

    def test_init_with_normalize(self, batch_size, size):
        matrix = np.random.random((batch_size, 2 * size, 2 * size))
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size), normalize=True)
        u, s, v = svd(sptm.matrix())
        np.testing.assert_allclose(
            s,
            np.ones_like(s),
            atol=1e-6,
        )

    def test_round(self, batch_size, size):
        matrix = np.random.random((batch_size, 2 * size, 2 * size))
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size))
        rounded = round(sptm, 3)
        assert isinstance(rounded, SingleParticleTransitionMatrixOperation)

    def test_real(self, batch_size, size):
        matrix = np.random.random((batch_size, 2 * size, 2 * size)).astype(complex)
        matrix += 1e-9j * np.random.random((batch_size, 2 * size, 2 * size))
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size))
        assert "complex" not in sptm.dtype
        real_sptm = sptm.real()
        assert isinstance(real_sptm, SingleParticleTransitionMatrixOperation)

    def test_trunc(self, batch_size, size):
        matrix = np.random.random((batch_size, 2 * size, 2 * size))
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size))
        truncated = sptm.__trunc__()
        assert isinstance(truncated, SingleParticleTransitionMatrixOperation)

    def test_check_is_in_so4_returns_false_non_det1(self, batch_size, size):
        matrix = 2.0 * np.eye(2 * size)
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size))
        assert not sptm.check_is_in_so4()

    def test_check_is_in_so4_returns_false_non_orthogonal(self, batch_size, size):
        matrix = np.random.random((2 * size, 2 * size))
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size))
        assert not sptm.check_is_in_so4()

    def test_from_operation_with_sptm_attribute(self, batch_size, size):
        n_qubits = 2
        mgo = MatchgateOperation.random(batch_size=batch_size, wires=np.arange(n_qubits), seed=size)

        class _OpWithSptm:
            wires = mgo.wires
            single_particle_transition_matrix = mgo.single_particle_transition_matrix

        sptm = SingleParticleTransitionMatrixOperation.from_operation(_OpWithSptm())
        assert isinstance(sptm, SingleParticleTransitionMatrixOperation)

    def test_check_matrix_invalid_raises(self, batch_size, size):
        matrix = 2.0 * np.eye(2 * size)
        with pytest.raises(ValueError):
            SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size), check_matrix=True)

    def test_clip_angles_scalar_shape(self, batch_size, size):
        scalar_angle = np.array(np.pi)
        clipped = SptmCompRyRy.clip_angles(scalar_angle)
        assert SptmCompRyRy.check_angles(clipped)

    def test_check_angles_invalid_not_equal_raises(self, batch_size, size):
        invalid_params = np.array([[0.5, 1.0]])
        with pytest.raises(ValueError):
            SptmCompRyRy.check_angles(invalid_params)

    def test_check_angles_invalid_equal_raises(self, batch_size, size):
        invalid_params = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError):
            SptmCompRyRy.check_angles(invalid_params)

    def test_pad_not_implemented_for_4d(self, batch_size, size):
        matrix = np.random.random((2, 2, 2 * size, 2 * size))
        sptm = SingleParticleTransitionMatrixOperation.__new__(SingleParticleTransitionMatrixOperation)
        sptm._matrix = matrix
        sptm._wires = np.arange(size)
        sptm._hyperparameters = {}
        from pennylane.wires import Wires

        with pytest.raises(NotImplementedError):
            sptm.pad(Wires(np.arange(size + 1)))

    def test_from_operation_fallback_raises(self, batch_size, size):
        from pennylane.wires import Wires

        class _InvalidOp:
            wires = Wires([0, 1])

            def matrix(self):
                return np.ones((4, 4))

        with pytest.raises(ValueError, match="Cannot convert"):
            SingleParticleTransitionMatrixOperation.from_operation(_InvalidOp())

    def test_check_matrix_valid_so4(self, batch_size, size):
        sptm_valid = SingleParticleTransitionMatrixOperation.random(wires=np.arange(size), seed=size)
        sptm_checked = SingleParticleTransitionMatrixOperation(
            sptm_valid.matrix(), wires=np.arange(size), check_matrix=True
        )
        assert sptm_checked is not None

    def test_check_is_in_so4_det1_but_not_orthogonal(self, batch_size, size):
        n = 2 * size
        matrix = np.eye(n, dtype=float)
        matrix[0, 1] = 1.0
        sptm = SingleParticleTransitionMatrixOperation(matrix=matrix, wires=np.arange(size))
        assert not sptm.check_is_in_so4()

    def test_matmul_non_sptm_raises(self, batch_size, size):
        sptm = SingleParticleTransitionMatrixOperation.random(wires=np.arange(size), seed=0)
        with pytest.raises(ValueError):
            _ = sptm @ "invalid"
