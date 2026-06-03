import numpy as np
import pennylane as qml
import pytest
from pennylane.ops.qubit import BasisStateProjector
from pennylane.wires import Wires

from matchcake import utils
from matchcake.devices.probability_strategies import CliffordSumStrategy
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.operations.state_preparation.state_prep_from_gates import StatePrepFromGates
from tests.configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestCliffordSumStrategy:
    @pytest.fixture
    def strategy(self):
        return CliffordSumStrategy()

    @pytest.mark.parametrize(
        "num_wires, seed",
        [(num_wires, seed) for num_wires, seeds in [(2, range(3)), (3, range(1))] for seed in seeds],
    )
    def test_sptm_unitary_probabilities(self, strategy, num_wires, seed):
        rn_gen = np.random.RandomState(seed=seed)

        system_state = rn_gen.choice([0, 1], size=num_wires)
        target_state = rn_gen.choice([0, 1], size=num_wires)
        wires = np.arange(len(system_state))
        state_prep_op = qml.BasisState(system_state, wires)
        global_sptm = SingleParticleTransitionMatrixOperation.random(wires=wires, seed=seed)

        @qml.qnode(qml.device("default.qubit", wires=wires))
        def ground_truth_circuit():
            state_prep_op.queue()
            global_sptm.to_qubit_operation()
            return qml.expval(BasisStateProjector(target_state, wires=wires))

        nif_probs = strategy(
            system_state=system_state,
            state_prep_op=state_prep_op,
            target_binary_states=target_state,
            wires=wires,
            all_wires=wires,
            global_sptm=global_sptm.matrix(),
            transition_matrix=utils.make_transition_matrix_from_action_matrix(global_sptm.matrix()),
        )

        qubit_probs = ground_truth_circuit()
        np.testing.assert_allclose(
            nif_probs.squeeze(),
            qubit_probs.squeeze(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_wires_as_int(self, strategy):
        num_wires = 2
        wires = np.arange(num_wires)
        global_sptm = SingleParticleTransitionMatrixOperation.random(wires=wires, seed=42)
        transition_matrix = utils.make_transition_matrix_from_action_matrix(global_sptm.matrix())
        state_prep_op = qml.BasisState(np.zeros(num_wires, dtype=int), wires)
        result = strategy(
            state_prep_op=state_prep_op,
            target_binary_states=np.array([0]),
            wires=0,
            all_wires=Wires(wires),
            transition_matrix=transition_matrix,
        )
        assert result is not None

    def test_compute_clifford_expvals_state_prep_from_gates(self):
        state_prep = StatePrepFromGates(lambda wires: [qml.Hadamard(wires=wires[0])], wires=[0, 1])
        indexes_shape = (4, 4)
        result = CliffordSumStrategy.compute_clifford_expvals(state_prep, indexes_shape)
        assert result.shape == indexes_shape
