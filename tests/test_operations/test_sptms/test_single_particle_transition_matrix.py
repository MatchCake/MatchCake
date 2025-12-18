import numpy as np
import pytest
import torch
from matchcake.operations import MatchgateOperation
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils import torch_utils

from ...configs import (
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
        torch.testing.assert_close(phased_identity, target_phased_identity, msg=f"exp(i * p) != U^(-1) V")

    def test_to_qubit_unitary(self, batch_size, size):
        sptm = SingleParticleTransitionMatrixOperation.random(batch_size=batch_size, wires=np.arange(size), seed=size)
        unitary = sptm.to_qubit_unitary()
        assert unitary._unitary_check(unitary.matrix(), int(2 ** len(unitary.wires)))
