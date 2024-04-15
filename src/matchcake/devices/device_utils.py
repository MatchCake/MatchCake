from typing import Iterable, Union, Optional

from ..operations.matchgate_operation import MatchgateOperation, _SingleParticleTransitionMatrix


class _VHMatchgatesContainer:
    def __init__(self):
        self.op_container = {}
        self.wires_set = set({})

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.op_container)

    def add(self, op: MatchgateOperation):
        if op.wires in self.op_container:
            self.op_container[op.wires] = self.op_container[op.wires] @ op
        else:
            self.op_container[op.wires] = op
        self.wires_set.update(op.wires.labels)

    def try_add(self, op: MatchgateOperation) -> bool:
        if op.wires in self.op_container:
            try:
                self.add(op)
            except Exception:
                return False
            return True
        is_wire_in_container = any([w in self.wires_set for w in op.wires.labels])
        if not is_wire_in_container:
            self.add(op)
            return True
        return False

    def extend(self, ops: Iterable[MatchgateOperation]) -> None:
        for op in ops:
            self.add(op)

    def try_extend(self, ops: Iterable[MatchgateOperation]) -> int:
        for i, op in enumerate(ops):
            if not self.try_add(op):
                return i
        return -1

    def clear(self):
        self.op_container.clear()
        self.wires_set.clear()

    def contract(self) -> Optional[Union[MatchgateOperation, _SingleParticleTransitionMatrix]]:
        if len(self) == 0:
            return None
        if len(self) == 1:
            return next(iter(self.op_container.values()))
        return _SingleParticleTransitionMatrix.from_operations(self.op_container.values())
