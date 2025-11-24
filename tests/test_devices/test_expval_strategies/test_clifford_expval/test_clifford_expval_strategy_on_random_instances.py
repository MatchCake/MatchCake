import numpy as np
import pennylane as qml
import pytest

from matchcake import NIFDevice
from matchcake.circuits import (
    RandomMatchgateHaarOperationsGenerator,
    RandomMatchgateOperationsGenerator,
)
from matchcake.devices.expval_strategies.clifford_expval._pauli_map import (
    _MAJORANA_COEFFS_MAP,
)
from matchcake.devices.expval_strategies.clifford_expval.clifford_expval_strategy import (
    CliffordExpvalStrategy,
)
from matchcake.operations import CompHH

from ....configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


@pytest.mark.parametrize("n_qubits, seed", [(n, s) for n in range(2, 6) for s in range(3)])
class TestCliffordExpvalStrategyOnRandomInstances:
    @pytest.fixture
    def strategy(self):
        return CliffordExpvalStrategy()

    @pytest.fixture
    def nif_device(self, n_qubits):
        return NIFDevice(wires=n_qubits)

    @pytest.fixture
    def qubit_device(self, n_qubits):
        return qml.device("default.qubit", wires=n_qubits)

    @pytest.fixture
    def rn_gen(self, seed):
        return np.random.RandomState(seed=seed)

    @pytest.fixture
    def random_hamiltonian(self, nif_device, rn_gen):
        hamiltonian_ops = [
            getattr(qml, p[0])(w0) @ getattr(qml, p[1])(w1)
            for w0, w1 in zip(nif_device.wires[:-1], nif_device.wires[1:])
            for p in _MAJORANA_COEFFS_MAP.keys()
        ]
        hamiltonian_coeffs = rn_gen.random(len(hamiltonian_ops))
        hamiltonian = qml.Hamiltonian(hamiltonian_coeffs, hamiltonian_ops)
        return hamiltonian

    @pytest.fixture
    def random_op_gen(self, nif_device, seed):
        return RandomMatchgateOperationsGenerator(
            # return RandomMatchgateHaarOperationsGenerator(
            wires=nif_device.wires,
            seed=seed,
        )

    def test_eye_sptm(self, strategy, n_qubits, qubit_device, rn_gen, random_hamiltonian):
        sptm = np.eye(2 * n_qubits)
        initial_state = rn_gen.choice([0, 1], size=n_qubits)
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_random_sptm_zeros_init(
        self,
        strategy,
        rn_gen,
        qubit_device,
        random_op_gen,
        random_hamiltonian,
        nif_device,
    ):
        initial_state = np.zeros(random_op_gen.n_qubits, dtype=int)
        random_op_gen.initial_state = initial_state
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)
        assert strategy.can_execute(state_prep_op, random_hamiltonian)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            for op in random_op_gen:
                op.queue()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        nif_device.reset()
        sptm = nif_device.apply_generator(random_op_gen).global_sptm.matrix()
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_op_gen.tolist() = }, {random_hamiltonian.terms() = }",
        )

    def test_random_sptm_ones_init(self, strategy, rn_gen, qubit_device, random_op_gen, random_hamiltonian, nif_device):
        initial_state = np.ones(random_op_gen.n_qubits, dtype=int)
        random_op_gen.initial_state = initial_state
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)
        assert strategy.can_execute(state_prep_op, random_hamiltonian)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            for op in random_op_gen:
                op.queue()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        nif_device.reset()
        sptm = nif_device.apply_generator(random_op_gen).global_sptm.matrix()
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_op_gen.tolist() = }, {random_hamiltonian.terms() = }",
        )

    def test_random_sptm(self, strategy, rn_gen, qubit_device, random_op_gen, random_hamiltonian, nif_device):
        initial_state = random_op_gen.get_initial_state(rn_gen)
        state_prep_op = qml.BasisState(initial_state, qubit_device.wires)
        assert strategy.can_execute(state_prep_op, random_hamiltonian)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            for op in random_op_gen:
                op.queue()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        nif_device.reset()
        sptm = nif_device.apply_generator(random_op_gen).global_sptm.matrix()
        # circuit = random_op_gen.get_ops()
        # sptm = circuit[1].to_sptm_operation()
        # for op in circuit[2:]:
        #     sptm = sptm @ op.to_sptm_operation()
        # sptm = sptm.matrix()
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_op_gen.tolist() = }, {random_hamiltonian.terms() = }",
        )
