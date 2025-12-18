import numpy as np
import pennylane as qml
import pytest
from pennylane.pauli import string_to_pauli_word
import torch

from matchcake import NIFDevice, MatchgateOperation
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
from matchcake.operations import CompHH, SingleParticleTransitionMatrixOperation, Rxx, Rzz, CompRyRy, CompRzRz, fSWAP, \
    FermionicSuperposition, CompRxRx
from matchcake.utils.majorana import majorana_to_pauli, MajoranaGetter
from matchcake.utils.torch_utils import to_tensor

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
            wires=nif_device.wires,
            seed=seed,
            op_types=[
                # TODO: Fail
                # MatchgateOperation,
                # CompRxRx,
                # CompHH,
                # CompRyRy,
                # FermionicSuperposition,
                # Rxx,

                # Pass
                Rzz,
                CompRzRz,
                fSWAP,
            ]
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
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_op_gen.tolist() = }, {random_hamiltonian.terms() = }",
        )

    @pytest.mark.parametrize("hamiltonian_term", ["XX", "YY", "XY", "YX"])
    def test_random_sptm_on_specific_hamiltonian(self, hamiltonian_term, strategy, rn_gen, qubit_device, random_op_gen, nif_device):
        random_hamiltonian = qml.Hamiltonian([1.0], [string_to_pauli_word(hamiltonian_term)])

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
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=sptm)
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{random_op_gen.tolist() = }, {random_hamiltonian.terms() = }",
        )

    def test_compute_clifford_expvals_rn_init(self, n_qubits, rn_gen, qubit_device, strategy):
        wires = np.arange(n_qubits).tolist()
        state_prep_op = qml.BasisState(rn_gen.choice([0, 1], n_qubits), wires)
        triu_indices = np.triu_indices(2 * n_qubits, k=1)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            return [qml.expval(majorana_to_pauli(mu) @ majorana_to_pauli(nu)) for mu, nu in zip(*triu_indices)]

        expvals = strategy.compute_clifford_expvals(state_prep_op)
        targets = ground_truth_circuit()
        np.testing.assert_allclose(expvals, targets)

    def test_compute_sum(self, rn_gen, n_qubits, strategy):
        majorana_indices = np.arange(2 * n_qubits).reshape(-1, 2)
        majorana_coeffs = np.ones(majorana_indices.shape[0]) * (-1j) ** np.arange(majorana_indices.shape[0])
        coeffs = np.arange(majorana_indices.shape[0]) / majorana_indices.shape[0]

        triu_indices = np.triu_indices(2 * n_qubits, k=1)
        h = np.zeros((2 * n_qubits, 2 * n_qubits))
        h[triu_indices] = np.arange(triu_indices[0].size) + 1
        h[triu_indices[1], triu_indices[0]] = -h[triu_indices]
        global_sptm = to_tensor(qml.math.expm(4 * h), dtype=torch.complex128)

        state_prep_op = qml.BasisState(rn_gen.choice([0, 1], size=n_qubits), np.arange(n_qubits).tolist())
        expvals = strategy._compute_full_clifford_expvals(state_prep_op, global_sptm)

        target_result = 0
        for k, i, j in np.ndindex((majorana_indices.shape[0], 2 * n_qubits, 2 * n_qubits)):
            target_result += (
                    majorana_coeffs[k]
                    * coeffs[k]
                    * global_sptm[..., majorana_indices[k, 0], i]
                    * global_sptm[..., majorana_indices[k, 1], j]
                    * expvals[i, j]
            )

        result = strategy._compute_sum(
            majorana_coeffs=majorana_coeffs,
            coeffs=coeffs,
            global_sptm=global_sptm,
            majorana_indices=majorana_indices,
            expvals=expvals,
        )
        np.testing.assert_allclose(result, target_result)

    def test_random_sptm_unitary(self, strategy, n_qubits, rn_gen, qubit_device, random_hamiltonian):
        wires = np.arange(n_qubits).tolist()
        global_sptm = SingleParticleTransitionMatrixOperation.random(wires=wires)
        state_prep_op = qml.BasisState(rn_gen.choice([0, 1], size=n_qubits), wires=wires)

        @qml.qnode(qubit_device)
        def ground_truth_circuit():
            state_prep_op.queue()
            global_sptm.to_qubit_unitary().queue()
            return qml.expval(random_hamiltonian)

        ground_truth_energy = ground_truth_circuit()
        clifford_energy = strategy(state_prep_op, random_hamiltonian, global_sptm=global_sptm.matrix())
        np.testing.assert_allclose(
            clifford_energy,
            ground_truth_energy,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"{state_prep_op = }"
        )
