from pennylane.wires import Wires

from ...operations.matchgate_operation import (
    MatchgateOperation,
    SingleParticleTransitionMatrixOperation,
)
from ...utils.math import circuit_matmul
from ..device_utils import circuit_or_fop_matmul
from .contraction_container import (
    _ContractionMatchgatesContainer,
    _ContractionMatchgatesContainerAddException,
)
from .contraction_strategy import ContractionStrategy


class _VHMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = op.cs_wires
        if wires in self.op_container:
            self.op_container[wires] = circuit_or_fop_matmul(first_matrix=self.op_container[wires], second_matrix=op)
            return True

        is_any_wire_in_container = any([w in self.wires_set for w in wires.labels])
        if is_any_wire_in_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not compatible with container."
            )
        self.op_container[wires] = op
        self.wires_set.update(wires.labels)
        return True


class NeighboursContractionStrategy(ContractionStrategy):
    NAME: str = "Neighbours"

    def get_container(self):
        return _VHMatchgatesContainer()
