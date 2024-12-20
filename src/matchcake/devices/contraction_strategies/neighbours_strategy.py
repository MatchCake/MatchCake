from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations.matchgate_operation import MatchgateOperation
from pennylane.wires import Wires


class _VHMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = Wires(sorted(op.wires))
        if wires in self.op_container:
            new_op = op @ self.op_container[wires]
            self.op_container[wires] = new_op
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
