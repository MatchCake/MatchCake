from copy import deepcopy

import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import (
    fSWAP, SptmRzRz, SptmFSwapRzRz, SptmFSwap, SptmFHH,
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
        # contraction_strategy=None,
        contraction_strategy="neighbours",
        # contraction_strategy="vertical",
        # contraction_strategy="horizontal",
        # contraction_strategy="forward",
    )
    nif_device_none = mc.NIFDevice(
        wires=wires,
        # contraction_strategy=None,
        contraction_strategy="neighbours",
    )

    def circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        for op in FermionicSuperposition(wires=wires).decomposition():
            yield op
        return

    def sptm_circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        for op in SptmFermionicSuperposition(wires=wires).decomposition():
            yield op
        return

    nif_device.execute_generator(circuit_gen(), apply=True, reset=True, cache_global_sptm=True)
    global_sptm = deepcopy(nif_device.apply_metadata["global_sptm"])

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
def test_sptm_fermionic_superposition(wires, is_odd):
    initial_state = np.zeros(len(wires))
    initial_state[0] = int(is_odd)

    nif_device = mc.NIFDevice(
        wires=wires,
        contraction_strategy="neighbours",
    )
    nif_device_none = mc.NIFDevice(
        wires=wires,
        contraction_strategy="neighbours",
    )

    def circuit_gen():
        yield qml.BasisState(initial_state, wires=wires)
        for op in FermionicSuperposition(wires=wires).decomposition():
            yield op
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
