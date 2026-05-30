import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake.devices.probability_strategies import ExplicitSumStrategy, LookupTableStrategy


class TestProbabilityStrategyBase:
    @pytest.fixture
    def strategy(self):
        return LookupTableStrategy()

    def test_check_required_kwargs_missing_raises(self, strategy):
        with pytest.raises(ValueError, match="Missing required keyword argument"):
            strategy.check_required_kwargs({})

    def test_batch_call_shape_mismatch_raises(self, strategy):
        state_prep = qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1])
        target_binary_states = np.array([[0, 0], [1, 0]])
        batch_wires = Wires([0, 1])
        with pytest.raises(ValueError):
            strategy.batch_call(
                state_prep_op=state_prep,
                target_binary_states=target_binary_states,
                batch_wires=batch_wires,
                lookup_table=None,
                pfaffian_method="det",
            )

    def test_base_batch_call_shape_mismatch_raises(self):
        # ExplicitSumStrategy uses the base ProbabilityStrategy.batch_call, so this
        # exercises the shape-mismatch guard in the base implementation directly.
        strategy = ExplicitSumStrategy()
        with pytest.raises(ValueError, match="does not match"):
            strategy.batch_call(
                state_prep_op=qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1]),
                target_binary_states=np.array([[0, 0], [1, 0]]),
                batch_wires=Wires([0, 1]),
            )
