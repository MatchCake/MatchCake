import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import utils
from matchcake.operations import (
    FermionicRotation,
    MatchgateOperation,
    fH,
    fRXX,
    fRYY,
    fRZZ,
    fSWAP,
)
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmFHH,
    SptmfRxRx,
    SptmFSwap,
    SptmIdentity,
    SptmRyRy,
    SptmRzRz,
)
from matchcake.utils import (
    MajoranaGetter,
    make_single_particle_transition_matrix_from_gate,
    recursive_kron,
    torch_utils,
)
from matchcake.utils.math import circuit_matmul

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "active_wire0, n_wires",
    [
        (active_wire0, n_wires)
        for n_wires in np.arange(2, 6)
        for active_wire0 in np.arange(n_wires - 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_matchgate_to_sptm_with_padding(active_wire0, n_wires):
    all_wires = qml.wires.Wires(list(range(n_wires)))
    mg = MatchgateOperation.random(wires=qml.wires.Wires([active_wire0, active_wire0 + 1]))
    padded_sptm = mg.to_sptm_operation().pad(wires=all_wires).matrix()

    # compute the sptm from the matchgate explicitly using tensor products
    mg_matrix = mg.matrix()
    u_ops = []
    for wire in all_wires:
        if wire == active_wire0:
            u_ops.append(mg_matrix)
        elif wire == active_wire0 + 1:
            pass
        else:
            u_ops.append(np.eye(2))
    u = recursive_kron(u_ops)
    sptm_from_u = make_single_particle_transition_matrix_from_gate(u)

    np.testing.assert_allclose(
        padded_sptm.squeeze(),
        sptm_from_u.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_sptm_trivial_pad_gradient_check(matrix):
    def func(p):
        wires = np.arange(0, p.shape[-1] // 2, dtype=int)
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=wires)
        new_op = op.pad(wires)
        return new_op.matrix()

    assert torch.autograd.gradcheck(
        func,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2 * size, 2 * size))
        for batch_size in [1, 4]
        for size in np.arange(2, 2 + N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_sptm_pad_gradient_check(matrix):
    w0 = np.random.randint(0, 10)
    new_wf = np.random.randint(1, 10)
    wires = np.arange(w0, w0 + matrix.shape[-1] // 2, dtype=int)
    new_wires = np.arange(0, np.max(wires) + new_wf, dtype=int)

    def func(p):
        op = SingleParticleTransitionMatrixOperation(matrix=p, wires=wires)
        new_op = op.pad(new_wires)
        return new_op.matrix()

    assert torch.autograd.gradcheck(
        func,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )
