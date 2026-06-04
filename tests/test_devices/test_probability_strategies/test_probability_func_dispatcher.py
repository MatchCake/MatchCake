import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake.devices.probability_strategies import (
    LookupTableStrategy,
    ProbabilityFuncDispatcher,
    ProductStateProbabilityStrategy,
)


class TestProbabilityFuncDispatcher:
    @pytest.fixture
    def basis_state(self):
        return qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1])

    def test_raises_when_no_strategy_can_execute(self, basis_state):
        # ProductStateProbabilityStrategy only handles ProductState inputs, so a plain
        # BasisState leaves the dispatcher without an applicable strategy.
        dispatcher = ProbabilityFuncDispatcher([ProductStateProbabilityStrategy()])
        with pytest.raises(ValueError, match="No probability strategy can execute"):
            dispatcher(
                state_prep_op=basis_state,
                target_binary_states=np.array([0, 0]),
                wires=Wires([0, 1]),
            )

    def test_can_execute_false_when_no_strategy_matches(self, basis_state):
        dispatcher = ProbabilityFuncDispatcher([ProductStateProbabilityStrategy()])
        assert dispatcher.can_execute(basis_state) is False

    def test_can_execute_true_when_a_strategy_matches(self, basis_state):
        dispatcher = ProbabilityFuncDispatcher([LookupTableStrategy()])
        assert dispatcher.can_execute(basis_state) is True

    def test_strategies_property_returns_the_given_list(self):
        strategies = [LookupTableStrategy()]
        dispatcher = ProbabilityFuncDispatcher(strategies)
        assert dispatcher.strategies is strategies
