import numpy as np
import pytest

import pennylane as qml
from matchcake import utils
from matchcake.operations import (
    fRXX,
    fRYY,
    fRZZ,
    FermionicRotation,
    fSWAP,
    fH,
    MatchgateOperation,
)
from matchcake.operations.single_particle_transition_matrices import (
    SptmfRxRx,
    SptmFSwap,
    SptmFHH,
    SptmIdentity,
    SptmRzRz,
    SptmRyRy,
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils import MajoranaGetter, recursive_kron, make_single_particle_transition_matrix_from_gate
from matchcake.utils.math import circuit_matmul
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)



@pytest.mark.parametrize(
    "active_wire0, n_wires",
    [
        (active_wire0, n_wires)
        for n_wires in np.arange(2, 6)
        for active_wire0 in np.arange(n_wires - 1)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
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
        padded_sptm.squeeze(), sptm_from_u.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


