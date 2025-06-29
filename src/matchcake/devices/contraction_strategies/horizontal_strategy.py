from pennylane.wires import Wires

from ...operations import SingleParticleTransitionMatrixOperation
from ...operations.matchgate_operation import MatchgateOperation
from ...utils.math import circuit_matmul
from ..device_utils import circuit_or_fop_matmul
from .contraction_container import (
    _ContractionMatchgatesContainer,
    _ContractionMatchgatesContainerAddException,
)
from .contraction_strategy import ContractionStrategy


class _HorizontalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = op.cs_wires
        if wires in self.op_container:
            new_op = circuit_or_fop_matmul(first_matrix=self.op_container[wires], second_matrix=op)
            self.op_container[wires] = new_op
        elif len(self) == 0:
            self.op_container[wires] = op
        else:
            raise _ContractionMatchgatesContainerAddException(f"Operation with wires {op.wires} not in container.")
        self.wires_set.update(wires.labels)
        return True


class HorizontalContractionStrategy(ContractionStrategy):
    NAME: str = "Horizontal"

    def get_container(self):
        return _HorizontalMatchgatesContainer()
