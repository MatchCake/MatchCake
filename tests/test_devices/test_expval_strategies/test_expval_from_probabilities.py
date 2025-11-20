import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake import NIFDevice
from matchcake.devices.expval_strategies.expval_from_probabilities import (
    ExpvalFromProbabilitiesStrategy,
)
from matchcake.operations import CompHH, CompZX

from ...configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestExpvalFromProbabilities:
    @pytest.fixture
    def strategy(self):
        return ExpvalFromProbabilitiesStrategy()

    @pytest.mark.parametrize(
        "circuit, hamiltonian",
        [
            (
                [CompHH(wires=[0, 1])],
                qml.Z(0) @ qml.Z(1),
            ),
            (
                [CompZX(wires=[0, 1])],
                qml.Z(0) @ qml.Z(1),
            ),
            (
                [CompZX(wires=[0, 1])],
                qml.Z(0) @ qml.I(1),
            ),
            (
                [CompZX(wires=[0, 1])],
                qml.I(0) @ qml.Z(1),
            ),
            (
                [CompZX(wires=[0, 1])],
                qml.I(0) @ qml.I(1),
            ),
        ],
    )
    def test_expval_on_circuits(self, circuit, hamiltonian, strategy):
        wires = Wires.all_wires([op.wires for op in circuit])
        qubit_device = qml.device("default.qubit", wires=wires)
        nif_device = NIFDevice(wires=wires)

        initial_state = np.zeros(len(qubit_device.wires))
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            for op in circuit:
                op.queue()
            return qml.expval(hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        nif_device.apply_generator(circuit)
        clifford_energy = strategy(state_prep_op, hamiltonian, prob=nif_device.probability(hamiltonian.wires))
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_call_on_something_cant_execute(self, strategy):
        with pytest.raises(ValueError):
            strategy(
                qml.BasisState([0, 0], [0, 1]),
                qml.Z(0) @ qml.X(1),
                prob=np.random.random(4),
            )
