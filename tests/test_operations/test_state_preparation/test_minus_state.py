import numpy as np
import pennylane as qml
import pytest
from pennylane import X

from matchcake import NonInteractingFermionicDevice
from matchcake.operations.state_preparation.minus_state import MinusState
from tests.configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestNIFDeviceMinusState:
    @pytest.mark.parametrize("num_wires", [2, 3, 4, 5])
    def test_state_vector(self, num_wires):
        qubit_dev = qml.device("lightning.qubit", wires=num_wires)

        def circuit():
            MinusState(wires=qubit_dev.wires)
            return qml.state()

        qubit_qnode = qml.QNode(circuit, qubit_dev)
        state = qubit_qnode()
        odd_binary_states = np.asarray([bin(i).count('1') % 2 for i in range(2 ** num_wires)])
        target_state = np.ones(2 ** num_wires) / np.sqrt(2 ** num_wires) * (-1) ** odd_binary_states
        np.testing.assert_allclose(state, target_state, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_vs_svs(self):
        pauli_string = [X(0) @ X(1)]
        nif_dev = NonInteractingFermionicDevice(wires=2)
        qubit_dev = qml.device("lightning.qubit", wires=2)

        def circuit():
            MinusState(wires=nif_dev.wires)
            return qml.expval(sum(pauli_string))

        nif_qnode = qml.QNode(circuit, nif_dev)
        qubit_qnode = qml.QNode(circuit, qubit_dev)
        expected_value = qubit_qnode()
        actual_value = nif_qnode()

        np.testing.assert_allclose(
            actual_value,
            expected_value,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
