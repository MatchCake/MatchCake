import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake.devices.probability_strategies import LookupTableStrategy


class TestProbabilityStrategyBase:
    @pytest.fixture
    def strategy(self):
        return LookupTableStrategy()

    def test_check_required_kwargs_missing_raises(self, strategy):
        with pytest.raises(ValueError, match="Missing required keyword argument"):
            strategy.check_required_kwargs({})

    def test_missing_required_kwarg_raises(self, strategy):
        state_prep = qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1])
        with pytest.raises(ValueError, match="Missing required keyword argument"):
            strategy(
                state_prep_op=state_prep,
                target_binary_states=np.array([0, 0]),
                wires=Wires([0, 1]),
                pfaffian_method="det",
                # lookup_table intentionally omitted
            )
