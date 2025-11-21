import numpy as np
import pennylane as qml
import pytest
from matchcake.circuits import RandomMatchgateHaarOperationsGenerator
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

    def get_random_hamiltonian(self, nif_device, rn_gen):
        hamiltonian_ops = [
            getattr(qml, p[0])(w0) @ getattr(qml, p[1])(w1)
            for w0, w1 in zip(nif_device.wires[:-1], nif_device.wires[1:])
            for p in ["ZZ", "ZI", "IZ", "II"]
        ]
        hamiltonian_coeffs = rn_gen.random(len(hamiltonian_ops))
        hamiltonian = qml.Hamiltonian(hamiltonian_coeffs, hamiltonian_ops)
        return hamiltonian

    @pytest.mark.parametrize("n_qubits, seed", [(n, s) for n in range(2, 6) for s in range(3)])
    def test_random_sptm(self, strategy, n_qubits, seed):
        nif_device = NIFDevice(wires=n_qubits)
        random_op_gen = RandomMatchgateHaarOperationsGenerator(
            wires=nif_device.wires,
            seed=seed,
        )
        qubit_device = qml.device("default.qubit", wires=n_qubits)
        rn_gen = np.random.RandomState(seed=seed)
        random_hamiltonian = self.get_random_hamiltonian(nif_device, rn_gen)

        initial_state = random_op_gen.get_initial_state(rn_gen)
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            for op in random_op_gen:
                op.queue()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        nif_device.apply_generator(random_op_gen)
        energy = strategy(state_prep_op, random_hamiltonian, prob=nif_device.probability())
        np.testing.assert_allclose(
            energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_op_gen.tolist() = }, {random_hamiltonian.terms() = }"
        )
