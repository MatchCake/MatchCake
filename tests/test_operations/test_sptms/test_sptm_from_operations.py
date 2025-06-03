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
from matchcake.utils import (
    MajoranaGetter,
    recursive_kron,
    make_single_particle_transition_matrix_from_gate,
)
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
        for n_wires in np.arange(4, 8)
        for active_wire0 in np.arange(n_wires - 3)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_two_matchgates_to_sptm_from_operations(active_wire0, n_wires):
    all_wires = qml.wires.Wires(list(range(n_wires)))
    mg0 = MatchgateOperation.random(wires=qml.wires.Wires([active_wire0, active_wire0 + 1]))
    all_wires_wo_mg0 = [w for w in all_wires if w not in list(mg0.wires.labels) + [active_wire0 - 1, n_wires - 1]]
    active_wire1 = np.random.choice(all_wires_wo_mg0)
    mg1 = MatchgateOperation.random(wires=qml.wires.Wires([active_wire1, active_wire1 + 1]))

    padded_sptm = SingleParticleTransitionMatrixOperation.from_operations([mg0, mg1]).pad(wires=all_wires).matrix()

    # compute the sptm from the matchgate explicitly using tensor products
    mg0_matrix = mg0.matrix()
    mg1_matrix = mg1.matrix()
    u_ops = []
    for wire in all_wires:
        if wire == active_wire0:
            u_ops.append(mg0_matrix)
        elif wire == active_wire0 + 1:
            pass
        elif wire == active_wire1:
            u_ops.append(mg1_matrix)
        elif wire == active_wire1 + 1:
            pass
        else:
            u_ops.append(np.eye(2))
    u = recursive_kron(u_ops)
    sptm_from_u = make_single_particle_transition_matrix_from_gate(u)
    np.testing.assert_equal(u.shape[-1], 2**n_wires)
    np.testing.assert_equal(padded_sptm.shape[-1], 2 * n_wires)
    np.testing.assert_allclose(
        padded_sptm.squeeze(),
        sptm_from_u.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
