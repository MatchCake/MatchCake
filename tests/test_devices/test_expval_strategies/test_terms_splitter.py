import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.expval_strategies.expval_from_probabilities import (
    ExpvalFromProbabilitiesStrategy,
)
from matchcake.devices.expval_strategies.terms_splitter import TermsSplitter


class TestTermsSplitter:
    @pytest.fixture
    def strategy(self):
        return TermsSplitter([ExpvalFromProbabilitiesStrategy()])

    def test_on_non_valid_observable(self, strategy):
        with pytest.raises(ValueError):
            strategy(
                qml.BasisState([0, 0], [0, 1]),
                qml.Z(0) @ qml.X(1),
                prob=np.random.random(4),
            )

    def test_format_observable(self, strategy):
        hamiltonian = strategy._format_observable(qml.X(0))
        assert isinstance(hamiltonian, qml.Hamiltonian)
        np.testing.assert_allclose(hamiltonian.coeffs, [1.0])

    def test_split_returns_per_strategy_lists(self, strategy):
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        observable = qml.Z(0)
        splits = strategy.split(state_prep_op, observable)
        assert len(splits) == 1
        assert len(splits[0]) >= 1

    def test_call_on_valid_observable(self, strategy):
        from matchcake import NIFDevice
        from matchcake.operations import CompZX

        nif_device = NIFDevice(wires=2)
        nif_device.apply_generator([CompZX(wires=[0, 1])])
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        nif_device.apply_state_prep(state_prep_op)
        observable = qml.Z(0) @ qml.I(1)
        result = strategy(state_prep_op, observable, prob_func=nif_device.probability)
        assert np.asarray(result).ndim <= 1
