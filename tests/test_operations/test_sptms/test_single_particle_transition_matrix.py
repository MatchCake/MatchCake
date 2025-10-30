import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.ops.qubit.observables import BasisStateProjector

from matchcake import NonInteractingFermionicDevice, utils
from matchcake.operations import FermionicRotation, fH, fRXX, fRYY, fRZZ, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmFHH,
    SptmfRxRx,
    SptmFSwap,
    SptmIdentity,
    SptmRyRy,
    SptmRzRz,
)
from matchcake.utils import torch_utils
from matchcake.utils.math import circuit_matmul

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
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
