from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations.matchgate_operation import MatchgateOperation


class _HorizontalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        if op.wires in self.op_container:
            new_op = op @ self.op_container[op.wires]
            self.op_container[op.wires] = new_op
        elif len(self) == 0:
            self.op_container[op.wires] = op
        else:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not in container."
            )
        self.wires_set.update(op.wires.labels)
        return True


class HorizontalContractionStrategy(ContractionStrategy):
    NAME: str = "Horizontal"

    def get_container(self):
        return _HorizontalMatchgatesContainer()
