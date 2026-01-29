import numpy as np
import pennylane as qml
import pytest
from pennylane.ops.qubit import BasisStateProjector

from matchcake import utils
from matchcake.devices.probability_strategies import CliffordSumStrategy
from matchcake.operations import SingleParticleTransitionMatrixOperation
from tests.configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestCliffordSumStrategy:
    @pytest.fixture
    def strategy(self):
        return CliffordSumStrategy()

    @pytest.mark.parametrize(
        "num_wires, seed",
        [(num_wires, seed) for seed in range(3) for num_wires in [2, 3]],
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
            target_binary_state=target_state,
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
