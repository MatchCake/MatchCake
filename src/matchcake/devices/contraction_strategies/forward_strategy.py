from pennylane.wires import Wires

from .contraction_strategy import ContractionStrategy
from .contraction_container import _ContractionMatchgatesContainer, _ContractionMatchgatesContainerAddException
from ...operations.matchgate_operation import MatchgateOperation
from ...operations.single_particle_transition_matrices import SingleParticleTransitionMatrixOperation


class _ForwardMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        wires = Wires(sorted(op.wires))
        op = SingleParticleTransitionMatrixOperation.from_operation(op)
        is_any_wire_in_container = any([w in self.wires_set for w in wires.labels])
        if is_any_wire_in_container:
            op_list = [
                self.op_container.pop(w)
                for w in list(self.op_container.keys())
                if any([lbl in wires.labels for lbl in w.labels])
            ]
            other = SingleParticleTransitionMatrixOperation.from_operations(op_list)
            new_op = other @ op
            self.op_container[new_op.wires] = new_op
            return True

        self.op_container[wires] = op
        self.wires_set.update(wires.labels)
        return True


class ForwardContractionStrategy(ContractionStrategy):
    NAME: str = "Forward"

    def get_container(self):
        return _ForwardMatchgatesContainer()
