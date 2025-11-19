import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.contraction_strategies import contraction_strategy_map

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import devices_init

set_seed(TEST_SEED)


hand_made_test_data = [
    ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 0], [qml.PauliZ(0) @ qml.Identity(1)]),
    ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 1], [qml.Identity(0) @ qml.PauliZ(1)]),
    ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([1, 1], [qml.PauliX(0) @ qml.PauliX(1)]),
    ([1, 0], [qml.PauliX(0) @ qml.PauliX(1)]),
    ([1, 0], [qml.PauliX(0) @ qml.PauliY(1)]),
    ([1, 1], [qml.PauliX(0) @ qml.PauliY(1)]),
    ([1, 1], [qml.PauliY(0) @ qml.PauliX(1)]),
    ([1, 1], [qml.PauliY(0) @ qml.PauliY(1)]),
    ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([0, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 0, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([0, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([0, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 0, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(1) @ qml.PauliZ(2)]),
    ([1, 1, 1], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(1) @ qml.PauliZ(2)]),
]

rn_test_data = [
    (
        np.random.choice([0, 1], size=n_wires),
        [p0(w) @ p1(w + np.random.randint(1, n_wires - w)) for w in np.random.randint(n_wires - 1, size=n_wires)],
        contraction_strategy,
    )
    for n_wires in range(2, 6)
    for i in range(N_RANDOM_TESTS_PER_CASE)
    for contraction_strategy in contraction_strategy_map.keys()
    for p0 in [qml.PauliX, qml.PauliY, qml.PauliZ]
    for p1 in [qml.PauliX, qml.PauliY, qml.PauliZ]
]


@pytest.mark.parametrize("basis_state,pauli_string", hand_made_test_data)
def test_nif_pauli_strings_on_basis_state_against_qubit_device(basis_state, pauli_string):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(sum(pauli_string))

    nif_device, qubit_device = devices_init(
        wires=len(basis_state),
        shots=None,
        contraction_strategy=None,
        name="lightning.qubit",
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = qubit_qnode()
    actual_value = nif_qnode()

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize("basis_state,pauli_string", hand_made_test_data)
def test_nif_pauli_strings_on_basis_state_against_qubit_device_outer_sum(basis_state, pauli_string):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return [qml.expval(pauli) for pauli in pauli_string]

    nif_device, qubit_device = devices_init(
        wires=len(basis_state),
        shots=None,
        contraction_strategy=None,
        name="lightning.qubit",
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = sum(qubit_qnode())
    actual_value = sum(nif_qnode())

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize("basis_state,pauli_string", hand_made_test_data)
def test_nif_pauli_strings_on_basis_state_against_qubit_device_probs(basis_state, pauli_string):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.probs()

    nif_device, qubit_device = devices_init(
        wires=len(basis_state),
        shots=None,
        contraction_strategy=None,
        name="lightning.qubit",
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = qubit_qnode()
    actual_value = nif_qnode()

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize("basis_state,pauli_string,contraction_strategy", rn_test_data)
def test_nif_pauli_strings_on_basis_state_against_qubit_device_rn_data(basis_state, pauli_string, contraction_strategy):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(sum(pauli_string))

    nif_device, qubit_device = devices_init(
        wires=len(basis_state),
        shots=None,
        contraction_strategy=contraction_strategy,
        name="lightning.qubit",
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = qubit_qnode()
    actual_value = nif_qnode()

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
