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
)
from matchcake.operations.single_particle_transition_matrices import (
    SptmRxRx,
    SptmFSwap,
    SptmFHH,
    SptmIdentity,
    SptmRzRz,
    SptmRyRy,
    SingleParticleTransitionMatrixOperation,
)
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)


def test_matchgate_equal_to_sptm_fswap():
    matchgate = fSWAP(wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmFSwap(wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wire0, wire1, all_wires",
    [
        (
                np.random.uniform(-np.pi, np.pi, batch_size).squeeze(),
                np.random.uniform(-np.pi, np.pi, batch_size).squeeze()
        )
        for batch_size in [1, 4]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_fswap_chain_equal_to_sptm_fswap(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRXX(params, wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRxRx(params, wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
