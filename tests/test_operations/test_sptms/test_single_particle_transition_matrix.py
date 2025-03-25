import numpy as np
import pytest

import pennylane as qml
import torch
from pennylane.ops.qubit.observables import BasisStateProjector

from matchcake import utils, NonInteractingFermionicDevice
from matchcake.operations import (
    fRXX,
    fRYY,
    fRZZ,
    FermionicRotation,
    fSWAP,
    fH,
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
from matchcake.utils import torch_utils
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
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRzRz.ALLOWED_ANGLES
        for phi in SptmRzRz.ALLOWED_ANGLES
    ] + [
        (np.full(batch_size, theta), np.full(batch_size, theta))
        for batch_size in [1, 4]
        for theta in SptmRzRz.EQUAL_ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_rzrz(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    params = SptmRzRz.clip_angles(params)
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
    ] + [
        (np.full(batch_size, theta), np.full(batch_size, theta))
        for batch_size in [1, 4]
        for theta in SptmRyRy.EQUAL_ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_ryry(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    params = SptmRyRy.clip_angles(params)
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
    "theta",
    [
        np.full(batch_size, theta)
        for batch_size in [1, 4]
        for theta in SptmRzRz.EQUAL_ALLOWED_ANGLES
    ]
)
def test_sptm_rzrz_is_so4_equal_angles(theta):
    params = np.asarray([theta, theta]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRzRz(params, wires=[0, 1])
    assert sptm_obj.check_is_in_so4()


@pytest.mark.parametrize(
    "theta, phi",
    [
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRyRy.ALLOWED_ANGLES
        for phi in SptmRyRy.ALLOWED_ANGLES
    ] + [
        (np.full(batch_size, theta), np.full(batch_size, theta))
        for batch_size in [1, 4]
        for theta in SptmRyRy.EQUAL_ALLOWED_ANGLES
    ]
)
def test_sptm_ryry_is_so4(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    sptm_obj = SptmRyRy(params, wires=[0, 1])
    assert sptm_obj.check_is_in_so4()


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
                np.random.uniform(-4*np.pi, 4*np.pi, batch_size).squeeze(),
                np.random.uniform(-4*np.pi, 4*np.pi, batch_size).squeeze()
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
                np.random.uniform(-4*np.pi, 4*np.pi, batch_size).squeeze(),
                np.random.uniform(-4*np.pi, 4*np.pi, batch_size).squeeze()
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
    "params",
    [
        [
            [np.pi, np.pi],
            [np.pi, 0],
            [0, np.pi],
            [0, 0],
        ],
        [np.pi, np.pi],
        [np.pi, 0],
        [0, np.pi],
        [0, 0],
    ]
)
def test_sptm_rzrz_is_so4_with_params(params):
    params = np.asarray(params).reshape(-1, 2).squeeze()
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
        (np.full(batch_size, theta), np.full(batch_size, phi))
        for batch_size in [1, 4]
        for theta in SptmRzRz.ALLOWED_ANGLES
        for phi in SptmRzRz.ALLOWED_ANGLES
    ]
)
def test_matchgate_equal_to_sptm_rzrz_adjoint(theta, phi):
    params = np.asarray([theta, phi]).reshape(-1, 2).squeeze()
    params = SptmRzRz.clip_angles(params)
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
    params = SptmRyRy.clip_angles(params)
    matchgate = fRYY(params, wires=[0, 1]).adjoint()
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmRyRy(params, wires=[0, 1]).adjoint().matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.random((batch_size, 2*size, 2*size))
        for batch_size in [1, 4]
        for size in np.arange(2, 2+N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_sptm_sum_gradient_check(matrix):
    def sptm_sum(p):
        return torch.sum(
            SingleParticleTransitionMatrixOperation(
                matrix=p,
                wires=np.arange(p.shape[-1] // 2)
            ).matrix()
        )

    torch.autograd.gradcheck(
        sptm_sum,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )


@pytest.mark.parametrize(
    "matrix, obs",
    [
        (np.random.random((batch_size, 2*size, 2*size)), obs)
        for batch_size in [1, 4]
        for size in np.arange(2, 2+N_RANDOM_TESTS_PER_CASE)
        for obs in [
            qml.PauliZ(0),
            sum([qml.PauliZ(i) for i in range(size)]),
            BasisStateProjector(np.zeros(size, dtype=int), wires=np.arange(size)),
        ]
    ]
)
def test_sptm_circuit_gradient_check(matrix, obs):
    nif_device = NonInteractingFermionicDevice(wires=matrix.shape[-1] // 2)

    def circuit(p):
        return nif_device.execute_generator(
            [SingleParticleTransitionMatrixOperation(matrix=p, wires=np.arange(p.shape[-1] // 2))],
            observable=obs,
            output_type="expval",
        )

    torch.autograd.gradcheck(
        circuit,
        torch_utils.to_tensor(matrix, torch.double).requires_grad_(),
        raise_exception=True,
        check_undefined_grad=False,
    )



