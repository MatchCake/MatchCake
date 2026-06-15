from unittest.mock import Mock

import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.probability_strategies import LookupTableStrategy


class TestLookupTableStrategy:
    @pytest.fixture
    def strategy(self):
        return LookupTableStrategy()

    def test_wires_as_int(self, strategy):
        lookup_table_mock = Mock()
        lookup_table_mock.return_value = np.array([[[0.0, 1.0], [-1.0, 0.0]]])  # (1, 2, 2)
        result = strategy(
            state_prep_op=qml.BasisState([0], wires=[0]),
            target_binary_states=np.array([0]),
            wires=0,
            lookup_table=lookup_table_mock,
        )
        assert result is not None

    def test_can_execute_basis_state_true(self, strategy):
        assert strategy.can_execute(qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1])) is True

    def test_can_execute_non_state_false(self, strategy):
        assert strategy.can_execute(qml.PauliX(0)) is False
