import itertools
from unittest.mock import Mock

import numpy as np
import pennylane as qml
import pytest

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.probability_strategies import LookupTableStrategy
from matchcake.operations import MatchgateOperation

from ...configs import ATOL_SCALAR_COMPARISON, set_seed


class TestLookupTableStrategy:
    @pytest.fixture
    def strategy(self):
        return LookupTableStrategy()

    @pytest.mark.parametrize("chunk_size", [1, 2, 100])
    def test_chunked_matches_unchunked(self, strategy, chunk_size):
        set_seed()
        n_wires = 3
        gate_seeds = [40 + idx for idx in range(n_wires - 1)]
        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            qml.BasisState(np.zeros(n_wires, dtype=int), wires=range(n_wires))
            for idx, seed in enumerate(gate_seeds):
                MatchgateOperation.random(wires=[idx, idx + 1], seed=seed)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        outcomes = np.array(list(itertools.product([0, 1], repeat=n_wires)))  # (8, 3)

        unchunked = np.asarray(nif_dev.get_states_probability(outcomes, nif_dev.wires))
        chunked = np.asarray(nif_dev.get_states_probability(outcomes, nif_dev.wires, pfaffian_chunk_size=chunk_size))
        np.testing.assert_allclose(chunked, unchunked, atol=ATOL_SCALAR_COMPARISON)

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
