import numpy as np
import pennylane as qml
import pytest
from pennylane import X, H

from matchcake import NonInteractingFermionicDevice
from matchcake.operations.state_preparation.state_prep_from_gates import StatePrepFromGates


class TestStatePrepFromGates:

    @pytest.mark.parametrize("gate, expval", [(X, 0), (H, 1)])
    def test_execute_with_expval(self, gate, expval):
        def gate_generator(wires):
            for wire in wires:
                yield gate(wire)

        def circuit():
            StatePrepFromGates(gate_generator, wires=[0, 1])
            return qml.expval(qml.X(0) @ qml.X(1))

        nif_dev = NonInteractingFermionicDevice(wires=2)
        nif_qnode = qml.QNode(circuit, nif_dev)
        pred_expval = nif_qnode()
        np.testing.assert_allclose(expval, pred_expval)

    @pytest.mark.parametrize(
        "gates, probs",
        [
            ([X, X, X, X], [0, 0, 0, 1]),
            ([X, qml.Identity, qml.Identity, X], [0, 0, 1, 0]),
            ([qml.Identity, qml.Identity, qml.Identity, qml.Identity], [1, 0, 0, 0])
        ]
    )
    def test_execute_with_prob(self, gates, probs):
        def gate_generator(wires):
            for wire in wires:
                yield gates[int(wire)](wire)

        def circuit():
            StatePrepFromGates(gate_generator, wires=[0, 1])
            return qml.probs()

        nif_dev = NonInteractingFermionicDevice(wires=2)
        nif_qnode = qml.QNode(circuit, nif_dev)
        pred_expval = nif_qnode()
        np.testing.assert_allclose(probs, pred_expval, atol=1e-5)
