import numpy as np
import pennylane as qml
import pytest
from pennylane import X, Y
from pennylane.wires import Wires

from matchcake import NIFDevice
from matchcake.devices.expval_strategies.clifford_expval.clifford_expval_strategy import (
    CliffordExpvalStrategy,
)
from matchcake.operations import (
    CompHH,
    CompZX,
    MatchgateIdentity,
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils import get_block_diagonal_matrix

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
            ),
            (
                [CompZX(wires=[0, 1]), CompZX(wires=[1, 2])],
                qml.X(0) @ qml.X(1),
            ),
            (
                [CompZX(wires=[0, 1]), CompZX(wires=[2, 3])],
                qml.X(0) @ qml.X(1),
            ),
            (
                [CompZX(wires=[0, 1]), CompZX(wires=[1, 2]), CompZX(wires=[2, 3])],
                qml.X(0) @ qml.X(1),
            ),
            (
                [
                    CompZX(wires=[0, 1]),
                    CompZX(wires=[1, 2]),
                    CompZX(wires=[0, 1]),
                    CompZX(wires=[2, 3]),
                    MatchgateIdentity(wires=[2, 3]),
                ],
                0.54 * qml.X(0) @ qml.X(1) + 0.71 * qml.Y(0) @ qml.Y(1),
            ),
            (
                [
                    CompZX(wires=[0, 1]),
                    CompZX(wires=[1, 2]),
                    CompZX(wires=[0, 1]),
                    CompZX(wires=[1, 2]),
                    MatchgateIdentity(wires=[2, 3]),
                ],
                0.54 * qml.X(0) @ qml.X(1),
            ),
            (
                [
                    CompZX(wires=[0, 1]),
                    CompZX(wires=[1, 2]),
                    CompZX(wires=[2, 3]),
                ],
                X(0) @ Y(1),
            ),
            (
                [
                    CompHH(wires=[0, 1]),
                    CompHH(wires=[1, 2]),
                ],
                X(0) @ X(1),
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
        sptm = nif_device.apply_generator(circuit).global_sptm.matrix()
        # sptm = SingleParticleTransitionMatrixOperation.from_operation(circuit[0])
        # for op in circuit[1:]:
        #     sptm = SingleParticleTransitionMatrixOperation.from_operation(op) @ sptm
        clifford_energy = strategy(state_prep_op, hamiltonian, global_sptm=sptm)
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
                global_sptm=np.eye(4),
            )

    @pytest.mark.parametrize("n_qubits", np.linspace(2, 10, num=8, dtype=int))
    def test_compute_clifford_expvals_zero_init(self, n_qubits, strategy):
        wires = np.arange(n_qubits).tolist()
        state_prep_op = qml.BasisState(np.zeros(n_qubits), wires)
        triu_indices = np.triu_indices(2 * n_qubits, k=1)

        expvals = strategy.compute_clifford_expvals(state_prep_op)
        targets = get_block_diagonal_matrix(n_qubits)[triu_indices]
        np.testing.assert_allclose(expvals, targets)

    @pytest.mark.parametrize("n_qubits", np.linspace(2, 10, num=8, dtype=int))
    def test_compute_clifford_expvals_ones_init(self, n_qubits, strategy):
        wires = np.arange(n_qubits).tolist()
        state_prep_op = qml.BasisState(np.ones(n_qubits), wires)
        triu_indices = np.triu_indices(2 * n_qubits, k=1)

        expvals = strategy.compute_clifford_expvals(state_prep_op)
        targets = get_block_diagonal_matrix(n_qubits).T[triu_indices]
        np.testing.assert_allclose(expvals, targets)

    @pytest.mark.parametrize(
        "hamiltonian, target",
        [
            (qml.Hamiltonian([1.0], [qml.X(0) @ qml.X(1)]), ["XX"]),
            (qml.Hamiltonian([1.0], [qml.X(1) @ qml.X(2)]), ["XX"]),
            (
                    qml.Hamiltonian(
                        [1, 1, 1, 1],
                        [
                            qml.X(1) @ qml.X(2),
                            qml.X(0) @ qml.Y(1),
                            qml.Y(0) @ qml.X(1),
                            qml.Y(0) @ qml.Y(1),
                        ]
                    ),
                    ["XX", "XY", "YX", "YY"]
            ),
            (
                    qml.Hamiltonian(
                        [1, 1, 1, 1],
                        [
                            qml.X(1) @ qml.X(2),
                            qml.X(2) @ qml.Y(3),
                            qml.Y(3) @ qml.X(4),
                            qml.Y(4) @ qml.Y(5),
                        ]
                    ),
                    ["XX", "XY", "YX", "YY"]
            ),
        ]
    )
    def test_hamiltonian_to_pauli_str(self, hamiltonian, target, strategy):
        output = strategy._hamiltonian_to_pauli_str(hamiltonian)
        assert output == target
