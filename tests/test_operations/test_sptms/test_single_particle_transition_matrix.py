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
        (np.full(batch_size, theta), np.full(batch_size, phi))
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
        rtol=1,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (np.full(batch_size, theta), np.full(batch_size, phi))
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
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRzRz.ALLOWED_ANGLES
        for phi in SptmRzRz.ALLOWED_ANGLES
    ]
)
def test_sptm_rzrz_is_so4(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRzRz(params, wires=[0, 1])
    assert sptm_obj.check_is_in_so4()


@pytest.mark.parametrize(
    "theta, phi",
    [
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRyRy.ALLOWED_ANGLES
        for phi in SptmRyRy.ALLOWED_ANGLES
    ]
)
def test_sptm_ryry_is_so4(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRyRy(params, wires=[0, 1])
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
def test_sptm_ryry_is_so4_out_of_angles(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRyRy(params, wires=[0, 1], check_angles=False, clip_angles=True)
    assert sptm_obj.check_is_in_so4()


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
def test_sptm_rzrz_is_so4_out_of_angles(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRzRz(params, wires=[0, 1], check_angles=False, clip_angles=True)
    assert sptm_obj.check_is_in_so4()


@pytest.mark.parametrize(
    "angles",
    [
        np.random.uniform(-10*np.pi, 10*np.pi, batch_size).squeeze()
        for batch_size in [1, 4, (4, 2)]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_ryry_clip_angles(angles):
    new_angles = SptmRyRy.clip_angles(angles)
    assert new_angles.shape == angles.shape
    assert type(new_angles) == type(angles)
    assert SptmRyRy.check_angles(new_angles)


@pytest.mark.parametrize(
    "angles",
    [
        np.random.uniform(-10*np.pi, 10*np.pi, batch_size).squeeze()
        for batch_size in [1, 4, (4, 2)]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_rzrz_clip_angles(angles):
    new_angles = SptmRzRz.clip_angles(angles)
    assert new_angles.shape == angles.shape
    assert type(new_angles) == type(angles)
    assert SptmRzRz.check_angles(new_angles)


def test_matchgate_equal_to_sptm_fswap_adjoint():
    matchgate = fSWAP(wires=[0, 1]).adjoint()
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmFSwap(wires=[0, 1]).adjoint().matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


def test_matchgate_equal_to_sptm_fhh_adjoint():
    matchgate = fH(wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmFHH(wires=[0, 1]).matrix()
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
def test_matchgate_equal_to_sptm_rxrx_adjoint(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRXX(params, wires=[0, 1]).adjoint()
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRxRx(params, wires=[0, 1]).adjoint().matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRzRz.ALLOWED_ANGLES
        for phi in SptmRzRz.ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_rzrz_adjoint(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRZZ(params, wires=[0, 1]).adjoint()
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRzRz(params, wires=[0, 1]).adjoint().matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=1,
    )


@pytest.mark.parametrize(
    "theta, phi",
    [
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRyRy.ALLOWED_ANGLES
        for phi in SptmRyRy.ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_ryry_adjoint(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    matchgate = fRYY(params, wires=[0, 1]).adjoint()
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRyRy(params, wires=[0, 1]).adjoint().matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
