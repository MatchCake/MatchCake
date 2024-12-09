from copy import deepcopy

import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import (
    fSWAP, SptmRzRz, SptmFSwapRzRz, SptmFSwap, SptmFHH, fH,
)
from matchcake.operations import FermionicSuperposition
import pennylane as qml

from matchcake.operations.single_particle_transition_matrices.sptm_fermionic_superposition import (
    SptmFermionicSuperposition,
)
from matchcake.utils import state_to_binary_string
from matchcake.utils.torch_utils import to_numpy
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    TEST_SEED,
)
from ...test_nif_device import devices_init

set_seed(TEST_SEED)


def test_sptm_fswap_matmul_fhh():
    wires = [0, 1]
    initial_state = np.zeros(len(wires))

    nif_device = mc.NIFDevice(
        wires=wires,
        contraction_strategy=None
    )
    nif_device_none = mc.NIFDevice(
        wires=wires,
        contraction_strategy=None,
    )

    def circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield fSWAP(wires=wires)
        yield fH(wires=wires)
        return

    def sptm_circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield SptmFSwap(wires=wires)
        yield SptmFHH(wires=wires)
        return

    nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"]).round(3)

    nif_device_none.execute_generator(sptm_circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    sptm_global_sptm = to_numpy(nif_device_none.apply_metadata["global_sptm"]).round(3)

    np.testing.assert_allclose(
        sptm_global_sptm, global_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


def test_sptm_fswap_matmul_fhh_probs():
    wires = [0, 1]
    initial_state = np.zeros(len(wires))
    nif_device = mc.NIFDevice(wires=wires, contraction_strategy="neighbours")

    def circuit():
        qml.BasisState(initial_state, wires=wires)
        fSWAP(wires=wires)
        fH(wires=wires)
        return qml.probs(wires=wires)

    def sptm_circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield SptmFSwap(wires=wires)
        yield SptmFHH(wires=wires)
        return

    qubit_device = qml.device("default.qubit", wires=len(wires), shots=None)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    qubit_probs = to_numpy(qubit_qnode()).squeeze()

    nif_probs = nif_device.execute_generator(sptm_circuit_gen(), output_type="probs", apply=True, reset=True)
    nif_probs = to_numpy(nif_probs).squeeze()

    np.testing.assert_allclose(
        nif_probs, qubit_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wires, is_odd",
    [
        (np.arange(n), is_odd)
        for n in range(2, 12)
        for is_odd in [False, True]
    ]
)
def test_sptm_fermionic_superposition_decomposition(wires, is_odd):
    initial_state = np.zeros(len(wires))
    initial_state[0] = int(is_odd)

    nif_device = mc.NIFDevice(
        wires=wires,
        contraction_strategy=None,
    )
    nif_device_none = mc.NIFDevice(
        wires=wires,
        contraction_strategy=None,
    )

    def circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield from FermionicSuperposition.compute_decomposition(wires=wires)
        return

    def sptm_circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield from SptmFermionicSuperposition.compute_decomposition(wires=wires)
        return

    nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"]).round(3)

    nif_device_none.execute_generator(sptm_circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    sptm_global_sptm = to_numpy(nif_device_none.apply_metadata["global_sptm"]).round(3)

    np.testing.assert_allclose(
        sptm_global_sptm, global_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wires, is_odd",
    [
        (np.arange(n), is_odd)
        for n in range(2, 12)
        for is_odd in [False, True]
    ]
)
def test_sptm_fermionic_superposition(wires, is_odd):
    initial_state = np.zeros(len(wires))
    initial_state[0] = int(is_odd)

    nif_device = mc.NIFDevice(
        wires=wires,
        contraction_strategy=None,
    )
    nif_device_none = mc.NIFDevice(
        wires=wires,
        contraction_strategy=None,
    )

    def circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield from FermionicSuperposition.compute_decomposition(wires=wires)
        return

    def sptm_circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield SptmFermionicSuperposition(wires=wires)
        return

    nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"]).round(3)

    nif_device_none.execute_generator(sptm_circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    sptm_global_sptm = to_numpy(nif_device_none.apply_metadata["global_sptm"])

    np.testing.assert_allclose(
        sptm_global_sptm, global_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wires, is_odd",
    [
        (np.arange(n), is_odd)
        for n in range(2, 12)
        for is_odd in [False, True]
    ]
)
def test_sptm_fermionic_superposition_probs(wires, is_odd):
    initial_state = np.zeros(len(wires))
    initial_state[0] = int(is_odd)

    nif_device = mc.NIFDevice(wires=wires, contraction_strategy="neighbours")

    def circuit():
        qml.BasisState(initial_state, wires=wires)
        FermionicSuperposition(wires=wires)
        return qml.probs(wires=wires)

    def sptm_circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        yield SptmFermionicSuperposition(wires=wires)
        return

    qubit_device = qml.device("default.qubit", wires=len(wires), shots=None)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    qubit_probs = to_numpy(qubit_qnode()).squeeze()

    nif_probs = nif_device.execute_generator(sptm_circuit_gen(), output_type="probs", apply=True, reset=True)
    nif_probs = to_numpy(nif_probs).squeeze()

    np.testing.assert_allclose(
        nif_probs, qubit_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wires",
    [
        np.arange(n)
        for n in range(2, 12)
    ]
)
def test_sptm_fermionic_superposition_unitary(wires):
    sptm = SptmFermionicSuperposition(wires=wires)
    assert sptm.check_is_unitary(atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)


