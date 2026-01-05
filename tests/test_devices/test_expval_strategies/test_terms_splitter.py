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
