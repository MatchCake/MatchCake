import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake.devices.contraction_strategies.contraction_container import (
    _ContractionMatchgatesContainer,
)
from matchcake.devices.contraction_strategies.forward_strategy import (
    _ForwardMatchgatesContainer,
)
from matchcake.operations import SptmCompRxRx


class TestContractionMatchgatesContainer:
    @pytest.fixture
    def container(self):
        return _ContractionMatchgatesContainer()

    @pytest.fixture
    def forward_container(self):
        return _ForwardMatchgatesContainer()

    def test_all_wires(self, container):
        assert container.all_wires == Wires(set())

    def test_keys(self, container):
        assert list(container.keys()) == []

    def test_sorted_values(self, container):
        assert container.sorted_values() == []

    def test_extend_adds_multiple_ops(self, forward_container):
        ops = [SptmCompRxRx(np.random.random(2), wires=[0, 1]) for _ in range(2)]
        forward_container.extend(ops)
        assert len(forward_container) > 0

    def test_contract_operations_with_non_matchgate_op(self, forward_container):
        identity = qml.Identity(0)
        matchgate = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        result = forward_container.contract_operations([identity, matchgate])
        assert len(result) >= 1

    def test_contract_operations_non_matchgate_clears_container(self, forward_container):
        matchgate = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        identity = qml.Identity(0)
        result = forward_container.contract_operations([matchgate, identity])
        assert len(result) >= 1

    def test_contract_operations_with_callback(self, forward_container):
        identity = qml.Identity(0)
        matchgate = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        callback_calls = []
        forward_container.contract_operations([matchgate, identity], callback=lambda i: callback_calls.append(i))
        assert len(callback_calls) >= 1
