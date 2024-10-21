from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations import MatchgateOperation, SingleParticleTransitionMatrixOperation


class _VerticalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        op = SingleParticleTransitionMatrixOperation.from_operation(op)
        if op.wires in self.op_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} already in container."
            )
        is_any_wire_in_container = any([w in self.wires_set for w in op.wires.labels])
        if is_any_wire_in_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not compatible with container."
            )
        self.op_container[op.wires] = op
        self.wires_set.update(op.wires.labels)
        return True


class VerticalContractionStrategy(ContractionStrategy):
    NAME: str = "Vertical"

    def get_container(self):
        return _VerticalMatchgatesContainer()
