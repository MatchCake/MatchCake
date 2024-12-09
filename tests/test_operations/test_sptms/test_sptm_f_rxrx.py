import numpy as np
import pytest

from matchcake.operations import (
    fRXX,
)
from matchcake.operations.single_particle_transition_matrices import (
    SptmfRxRx,
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
    "theta, phi",
    [
        (
                np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze(),
                np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        )
        for batch_size in [1, 4]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_matchgate_equal_to_sptm_f_rxrx(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    params = SptmfRxRx.clip_angles(params)
    matchgate = fRXX(params, wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmfRxRx(params, wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (
                np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze(),
                np.random.uniform(-4 * np.pi, 4 * np.pi, batch_size).squeeze()
        )
        for batch_size in [1, 4]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_matchgate_equal_to_sptm_f_rxrx_multiple(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2)
    params = SptmfRxRx.clip_angles(params)

    matchgates = [fRXX(p, wires=[0, 1]) for p in params]
    matchgate = matchgates[0]
    for mg in matchgates[1:]:
        matchgate = circuit_matmul(matchgate, mg)

    m_sptm = matchgate.single_particle_transition_matrix
    sptms = [SptmfRxRx(p, wires=[0, 1]).matrix() for p in params]
    sptm = sptms[0]
    for s in sptms[1:]:
        sptm = circuit_matmul(sptm, s, operator="einsum")
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (
                np.random.uniform(-np.pi, np.pi, batch_size).squeeze(),
                np.random.uniform(-np.pi, np.pi, batch_size).squeeze()
        )
        for batch_size in [1, 4]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_matchgate_equal_to_sptm_f_rxrx_adjoint(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRXX(params, wires=[0, 1]).adjoint()
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmfRxRx(params, wires=[0, 1]).adjoint().matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
