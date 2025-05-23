import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import (
    fSWAP,
    SptmRzRz,
    SptmFSwapRzRz,
)
from matchcake.operations import FermionicSuperposition
import pennylane as qml

from matchcake.utils import state_to_binary_string
from matchcake.utils.torch_utils import to_numpy
from ..configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    TEST_SEED,
)
from ..test_nif_device import devices_init

set_seed(TEST_SEED)


@pytest.mark.parametrize("wires", [np.arange(n) for n in range(2, 12)])
def test_fermionic_superposition_even_states(wires):
    nif_device, qubit_device = devices_init(wires=wires)

    def circuit():
        qml.BasisState(np.zeros(len(wires), dtype=int), wires=wires)
        # FermionicSuperposition(wires=wires)
        [op for op in FermionicSuperposition.compute_decomposition(wires=wires)]
        return qml.state()

    qubit_qnode = qml.QNode(circuit, qubit_device)
    qubit_state = to_numpy(qubit_qnode())

    # make sure that the state is in superposition of all even parity states
    binary_indexes = [
        state_to_binary_string(i, n=len(wires)) for i in range(2 ** len(wires))
    ]
    is_even_parity = [
        np.sum([int(bit) for bit in binary_index]) % 2 == 0
        for binary_index in binary_indexes
    ]
    # make that all even parity states have equal amplitude
    even_states = qubit_state[..., is_even_parity]
    amplitude = 1 / np.sqrt(even_states.shape[-1])
    np.testing.assert_allclose(
        even_states,
        amplitude,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize("wires", [np.arange(n) for n in range(2, 12)])
def test_fermionic_superposition_odd_states(wires):
    nif_device, qubit_device = devices_init(wires=wires)

    def circuit():
        initial_state = np.zeros(len(wires), dtype=int)
        initial_state[0] = 1
        qml.BasisState(initial_state, wires=wires)
        FermionicSuperposition(wires=wires)
        return qml.state()

    qubit_qnode = qml.QNode(circuit, qubit_device)
    qubit_state = to_numpy(qubit_qnode())

    # make sure that the state is in superposition of all even parity states
    binary_indexes = [
        state_to_binary_string(i, n=len(wires)) for i in range(2 ** len(wires))
    ]
    mask_parity = [
        np.sum([int(bit) for bit in binary_index]) % 2 != 0
        for binary_index in binary_indexes
    ]
    # make that all even parity states have equal amplitude
    roi_states = qubit_state[..., mask_parity]
    amplitude = 1 / np.sqrt(roi_states.shape[-1])
    np.testing.assert_allclose(
        roi_states,
        amplitude,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
