import pytest
from pennylane.wires import Wires

from matchcake.devices.contraction_strategies.contraction_container import (
    _ContractionMatchgatesContainer,
)


class TestContractionMatchgatesContainer:
    @pytest.fixture
    def container(self):
        return _ContractionMatchgatesContainer()

    def test_all_wires(self, container):
        assert container.all_wires == Wires(set())

    def test_keys(self, container):
        assert list(container.keys()) == []

    def test_sorted_values(self, container):
        assert container.sorted_values() == []
