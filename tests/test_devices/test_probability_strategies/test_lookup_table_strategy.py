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
            pfaffian_method="det",
        )
        assert result is not None
