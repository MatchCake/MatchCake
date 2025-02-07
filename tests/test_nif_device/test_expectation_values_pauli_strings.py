import numpy as np
import pennylane as qml
import pytest
from . import devices_init
from ..configs import (
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)

hand_made_test_data = [
    ([0, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([0, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([1, 0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ([1, 1], [qml.PauliZ(0) @ qml.PauliZ(1)]),
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


@pytest.mark.parametrize(
    "basis_state,pauli_string",
    hand_made_test_data
)
def test_nif_pauli_strings_on_basis_state_against_qubit_device(basis_state, pauli_string):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.expval(sum(pauli_string))

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=None, name="lightning.qubit"
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = qubit_qnode()
    actual_value = nif_qnode()

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON
    )


@pytest.mark.parametrize(
    "basis_state,pauli_string",
    hand_made_test_data
)
def test_nif_pauli_strings_on_basis_state_against_qubit_device_outer_sum(basis_state, pauli_string):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return [qml.expval(pauli) for pauli in pauli_string]

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=None, name="lightning.qubit"
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = sum(qubit_qnode())
    actual_value = sum(nif_qnode())

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON
    )


@pytest.mark.parametrize(
    "basis_state,pauli_string",
    hand_made_test_data
)
def test_nif_pauli_strings_on_basis_state_against_qubit_device_probs(basis_state, pauli_string):
    def circuit():
        qml.BasisState(np.asarray(basis_state), wires=np.arange(len(basis_state)))
        return qml.probs()

    nif_device, qubit_device = devices_init(
        wires=len(basis_state), shots=None, contraction_strategy=None, name="lightning.qubit"
    )
    nif_qnode = qml.QNode(circuit, nif_device)
    qubit_qnode = qml.QNode(circuit, qubit_device)
    expected_value = qubit_qnode()
    actual_value = nif_qnode()

    np.testing.assert_allclose(
        actual_value,
        expected_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON
    )
