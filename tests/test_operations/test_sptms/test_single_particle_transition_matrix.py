import numpy as np
import pytest

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
)
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
        (theta, phi)
        for batch_size in [1, 4]
        for theta in np.random.uniform(-np.pi, np.pi, batch_size)
        for phi in np.random.uniform(-np.pi, np.pi, batch_size)
    ]
)
def test_matchgate_equal_to_sptm_rxrx(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRXX(params, wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRxRx(params, wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (theta, phi)
        for batch_size in [1, 4]
        for theta in SptmRzRz.ALLOWED_ANGLES
        for phi in SptmRzRz.ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_rzrz(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRZZ(params, wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRzRz(params, wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (theta, phi)
        for batch_size in [1, 4]
        for theta in SptmRyRy.ALLOWED_ANGLES
        for phi in SptmRyRy.ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_ryry(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRYY(params, wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRyRy(params, wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (theta, phi)
        for batch_size in [1, 4]
        for theta in SptmRzRz.ALLOWED_ANGLES
        for phi in SptmRzRz.ALLOWED_ANGLES
    ]
)
def test_sptm_rzrz_is_so4(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRzRz(params, wires=[0, 1])
    sptm = sptm_obj.matrix()

    np.testing.assert_allclose(
        np.linalg.det(sptm), 1,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    np.testing.assert_allclose(
        np.linalg.inv(sptm), sptm.T,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    assert sptm_obj.check_is_in_so4()


@pytest.mark.parametrize(
    "theta, phi",
    [
        (theta, phi)
        for batch_size in [1, 4]
        for theta in SptmRyRy.ALLOWED_ANGLES
        for phi in SptmRyRy.ALLOWED_ANGLES
    ]
)
def test_sptm_ryry_is_so4(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRyRy(params, wires=[0, 1])
    sptm = sptm_obj.matrix()

    np.testing.assert_allclose(
        np.linalg.det(sptm), 1,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    np.testing.assert_allclose(
        np.linalg.inv(sptm), sptm.T,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    assert sptm_obj.check_is_in_so4()


def test_matchgate_equal_to_sptm_fswap():
    matchgate = fSWAP(wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmFSwap(wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


def test_matchgate_equal_to_sptm_fhh():
    matchgate = fH(wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmFHH(wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


def test_matchgate_equal_to_sptm_identity():
    sptm = SptmIdentity(wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, np.eye(4),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
