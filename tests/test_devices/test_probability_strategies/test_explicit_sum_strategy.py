import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake import utils
from matchcake.devices.probability_strategies import ExplicitSumStrategy
from matchcake.operations import SingleParticleTransitionMatrixOperation


class TestExplicitSumStrategy:
    @pytest.fixture
    def strategy(self):
        return ExplicitSumStrategy()

    @pytest.fixture
    def two_qubit_setup(self):
        num_wires = 2
        wires = np.arange(num_wires)
        global_sptm = SingleParticleTransitionMatrixOperation.random(wires=wires, seed=42)
        transition_matrix = utils.make_transition_matrix_from_action_matrix(global_sptm.matrix())
        state_prep_op = qml.BasisState(np.zeros(num_wires, dtype=int), wires)
        return wires, transition_matrix, state_prep_op

    def test_wires_as_int(self, strategy, two_qubit_setup):
        wires, transition_matrix, state_prep_op = two_qubit_setup
        result = strategy(
            state_prep_op=state_prep_op,
            target_binary_state=np.array([0]),
            wires=0,
            all_wires=Wires(wires),
            transition_matrix=transition_matrix,
        )
        assert result is not None
