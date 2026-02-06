import numpy as np
import pennylane as qml
import pytest
from pennylane.ops.qubit import BasisStateProjector
from pennylane.wires import Wires

from matchcake import BatchHamiltonian, NIFDevice
from matchcake.circuits import RandomMatchgateHaarOperationsGenerator
from matchcake.devices.expval_strategies.expval_from_probabilities import (
    ExpvalFromProbabilitiesStrategy,
)
from matchcake.operations import CompHH, CompZX, SingleParticleTransitionMatrixOperation

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
        energy = strategy(state_prep_op, hamiltonian, prob_func=nif_device.probability)
        np.testing.assert_allclose(
            energy,
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

    def test_call_on_projector_cant_execute(self, strategy):
        with pytest.raises(ValueError):
            strategy(
                qml.BasisState([0, 0], [0, 1]),
                BasisStateProjector([1, 0], wires=[0, 1]),
                prob=np.random.random(4),
            )

    def test_call_on_batch_hamiltonian(self, strategy):
        batch_hamiltonian = BatchHamiltonian(
            [0.1, 0.2, 0.3], [qml.Z(0) @ qml.Z(1), qml.Z(0) @ qml.I(1), qml.I(0) @ qml.Z(1)]
        )
        expvals = strategy(
            qml.BasisState([0, 0], [0, 1]), batch_hamiltonian, prob=np.broadcast_to(np.asarray([1, 0, 0, 0]), (3, 4))
        )

        @qml.qnode(qml.device("default.qubit", wires=2))
        def ground_truth_circuit():
            qml.BasisState([0, 0], [0, 1])
            return [qml.expval(c * h) for c, h in zip(batch_hamiltonian.coeffs, batch_hamiltonian.ops)]

        expected_expvals = ground_truth_circuit()
        np.testing.assert_allclose(
            expvals,
            expected_expvals,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
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
        rn_gen = np.random.RandomState(seed=seed)
        qubit_device = qml.device("default.qubit", wires=n_qubits)

        rn_global_sptm = SingleParticleTransitionMatrixOperation.random(np.arange(n_qubits), seed=seed)
        initial_state = rn_gen.choice([0, 1], size=n_qubits)
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)
        nif_device = NIFDevice(wires=n_qubits)
        nif_device.global_sptm = rn_global_sptm
        random_hamiltonian = self.get_random_hamiltonian(nif_device, rn_gen)
        nif_device.apply_state_prep(state_prep_op)
        energy = strategy(state_prep_op, random_hamiltonian, prob_func=nif_device.probability)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            nif_device.global_sptm.to_qubit_operation()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        np.testing.assert_allclose(
            energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_hamiltonian.terms() = }",
        )

    def test_long_range_hamiltonian(self, strategy):
        long_range_hamiltonian = qml.Hamiltonian(
            [0.5, -0.3, 0.8],
            [qml.Z(0) @ qml.Z(8), qml.Z(1) @ qml.I(7), qml.I(0) @ qml.Z(5)],
        )

        n_qubits = max(long_range_hamiltonian.wires) + 1
        qubit_device = qml.device("default.qubit", wires=n_qubits)

        initial_state = np.zeros(n_qubits, dtype=int)
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)
        nif_device = NIFDevice(wires=n_qubits)
        random_global_sptm = SingleParticleTransitionMatrixOperation.random(qubit_device.wires, seed=42)
        nif_device.global_sptm = random_global_sptm
        nif_device.apply_state_prep(state_prep_op)
        energy = strategy(
            state_prep_op,
            long_range_hamiltonian,
            prob_func=nif_device.probability,
        )

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            random_global_sptm.to_qubit_operation().queue()
            return qml.expval(long_range_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        np.testing.assert_allclose(
            energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
