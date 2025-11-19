import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake import NIFDevice
from matchcake.circuits import RandomMatchgateHaarOperationsGenerator
from matchcake.devices.expval_strategies.clifford_expval._pauli_map import (
    _MAJORANA_COEFFS_MAP,
)
from matchcake.devices.expval_strategies.clifford_expval.clifford_expval_strategy import (
    CliffordExpvalStrategy,
)
from matchcake.operations import CompHH, CompZX

from ....configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestCliffordExpvalStrategy:
    @pytest.fixture
    def strategy(self):
        return CliffordExpvalStrategy()

    @pytest.mark.parametrize(
        "circuit, hamiltonian",
        [
            (
                [CompHH(wires=[0, 1])],
                qml.X(0) @ qml.X(1),
            ),
            (
                    [CompZX(wires=[0, 1])],
                    qml.X(0) @ qml.X(1),
            )
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
        sptm = nif_device.apply_generator(circuit).global_sptm.matrix()
        clifford_energy = strategy(state_prep_op, hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
