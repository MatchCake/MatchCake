import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake.devices.probability_strategies import (
    LookupTableStrategy,
    ProbabilityStrategy,
)


class _ConstStrategy(ProbabilityStrategy):
    """Minimal concrete strategy exercising the base ``__call__`` and ``can_execute``.

    It scores a single outcome by its Hamming weight so the batched path returns a
    predictable vector, without depending on any device kwargs.
    """

    NAME = "ConstDummy"

    def _compute_single(self, *, state_prep_op, target_binary_state, wires, **kwargs):
        return qml.math.sum(qml.math.asarray(target_binary_state))


class TestProbabilityStrategyBase:
    @pytest.fixture
    def strategy(self):
        return LookupTableStrategy()

    def test_base_can_execute_defaults_to_true(self):
        basis = qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1])
        assert _ConstStrategy().can_execute(basis) is True

    def test_base_call_batch_broadcasts_1d_wires(self):
        strategy = _ConstStrategy()
        targets = np.array([[0, 1], [1, 1]])
        out = strategy(
            state_prep_op=qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1]),
            target_binary_states=targets,
            wires=Wires([0, 1]),
        )
        assert tuple(qml.math.shape(out)) == (2,)
        np.testing.assert_array_equal(np.asarray(out), [1, 2])

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
