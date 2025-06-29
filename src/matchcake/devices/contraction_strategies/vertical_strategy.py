from pennylane.wires import Wires

from ...operations import MatchgateOperation, SingleParticleTransitionMatrixOperation
from .contraction_container import (
    _ContractionMatchgatesContainer,
    _ContractionMatchgatesContainerAddException,
)
from .contraction_strategy import ContractionStrategy


class _VerticalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = op.cs_wires
        if wires in self.op_container:
            raise _ContractionMatchgatesContainerAddException(f"Operation with wires {op.wires} already in container.")
        is_any_wire_in_container = any([w in self.all_cs_wires for w in wires.labels])
        if is_any_wire_in_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not compatible with container."
            )
        self.op_container[wires] = op
        self.wires_set.update(wires.labels)
        return True


class VerticalContractionStrategy(ContractionStrategy):
    NAME: str = "Vertical"

    def get_container(self):
        return _VerticalMatchgatesContainer()
