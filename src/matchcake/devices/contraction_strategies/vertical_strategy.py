from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations import MatchgateOperation, SingleParticleTransitionMatrixOperation
from pennylane.wires import Wires


class _VerticalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = Wires(sorted(op.wires))
        op = SingleParticleTransitionMatrixOperation.from_operation(op)
        if op.wires in self.op_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} already in container."
            )
        is_any_wire_in_container = any([w in self.wires_set for w in wires.labels])
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
