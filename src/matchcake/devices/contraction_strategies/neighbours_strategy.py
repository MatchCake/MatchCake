from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations.matchgate_operation import MatchgateOperation


class _VHMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        if op.wires in self.op_container:
            new_op = op @ self.op_container[op.wires]
            self.op_container[op.wires] = new_op
            return True

        is_any_wire_in_container = any([w in self.wires_set for w in op.wires.labels])
        if is_any_wire_in_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not compatible with container."
            )
        self.op_container[op.wires] = op
        self.wires_set.update(op.wires.labels)
        return True


class NeighboursContractionStrategy(ContractionStrategy):
    NAME: str = "Neighbours"

    def get_container(self):
        return _VHMatchgatesContainer()
