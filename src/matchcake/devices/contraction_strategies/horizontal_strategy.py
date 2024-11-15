from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations.matchgate_operation import MatchgateOperation
from pennylane.wires import Wires


class _HorizontalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = Wires(sorted(op.wires))
        if wires in self.op_container:
            new_op = op @ self.op_container[wires]
            self.op_container[wires] = new_op
        elif len(self) == 0:
            self.op_container[wires] = op
        else:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not in container."
            )
        self.wires_set.update(wires.labels)
        return True


class HorizontalContractionStrategy(ContractionStrategy):
    NAME: str = "Horizontal"

    def get_container(self):
        return _HorizontalMatchgatesContainer()
