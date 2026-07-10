import numpy as np
import pennylane as qml
import pytest
from pennylane import H, X, Y

from matchcake import NonInteractingFermionicDevice
from matchcake.operations.state_preparation.state_prep_from_gates import (
    StatePrepFromGates,
)


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
            ([qml.Identity, qml.Identity, qml.Identity, qml.Identity], [1, 0, 0, 0]),
        ],
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

    def test_to_basis_state(self):
        gates = [X(0), X(1)]
        basis_state = StatePrepFromGates(lambda _: gates, wires=[0, 1]).to_basis_state()
        np.testing.assert_array_equal(basis_state.parameters[0], [1, 1])

    def test_to_bais_state_with_invalid_gates(self):
        gates = [X(0), Y(1)]
        with pytest.raises(ValueError):
            StatePrepFromGates(lambda _: gates, wires=[0, 1]).to_basis_state()

    def test_is_basis_state(self):
        gates = [X(0), X(1)]
        assert StatePrepFromGates(lambda _: gates, wires=[0, 1]).is_basis_state

        gates = [X(0), Y(1)]
        assert not StatePrepFromGates(lambda _: gates, wires=[0, 1]).is_basis_state
