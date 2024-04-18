import warnings
from typing import Iterable, Union, Optional, Literal, List

from ..operations.matchgate_operation import MatchgateOperation, _SingleParticleTransitionMatrix


class _ContractionMatchgatesContainerAddException(Exception):
    pass


class _ContractionMatchgatesContainer:
    def __init__(self):
        self.op_container = {}
        self.wires_set = set({})

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.op_container)

    def add(self, op: MatchgateOperation) -> bool:
        """
        Add an operation to the container. If the operation is not compatible with the operations already in the
        container, an exception will be raised.

        :raises _ContractionMatchgatesContainerAddException: If the operation is not compatible with the operations

        :param op: The operation to add.
        :type op: MatchgateOperation
        :return: True if the operation was added.
        :rtype: bool
        """
        raise NotImplementedError

    def try_add(self, op: MatchgateOperation) -> bool:
        try:
            return self.add(op)
        except _ContractionMatchgatesContainerAddException:
            return False
        except Exception as e:
            warnings.warn(f"Unexpected exception in try_add: {e}", RuntimeWarning)
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

    def contract(self) -> Optional[_SingleParticleTransitionMatrix]:
        if len(self) == 0:
            return None
        sptms = []
        for op in self.op_container.values():
            if isinstance(op, _SingleParticleTransitionMatrix):
                sptms.append(op)
            elif isinstance(op, MatchgateOperation):
                sptms.append(_SingleParticleTransitionMatrix(op.single_particle_transition_matrix, op.wires))
            else:
                raise TypeError(f"Unexpected type in container: {type(op)}")
        return _SingleParticleTransitionMatrix.from_spt_matrices(sptms)

    def push_contract(
            self,
            op: MatchgateOperation,
    ) -> Optional[Union[MatchgateOperation, _SingleParticleTransitionMatrix]]:
        """
        This method will try to add the operation to the container. If it can't, it will contract the operations in the
        container, return the contracted operation, clear the current container and add the new operation to it.

        :param op: The operation to add.
        :return: The contracted operation if the container was cleared, None otherwise.
        """
        if not self.try_add(op):
            contracted_op = self.contract()
            self.clear()
            self.add(op)
            return contracted_op
        return None

    def contract_operations(self, operations) -> List[Union[MatchgateOperation, _SingleParticleTransitionMatrix]]:
        new_operations = []
        for op in operations:
            if not isinstance(op, MatchgateOperation):
                new_operations.append(op)
                if self:
                    new_operations.append(self.contract())
                    self.clear()
                continue
            new_op = self.push_contract(op)
            if new_op is not None:
                new_operations.append(new_op)
        if self:
            new_operations.append(self.contract())
            self.clear()
        return new_operations


class _VerticalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        op = _SingleParticleTransitionMatrix(op.single_particle_transition_matrix, op.wires)
        if op.wires in self.op_container:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} already in container."
            )
        else:
            self.op_container[op.wires] = op
        self.wires_set.update(op.wires.labels)
        return True


class _HorizontalMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        # TODO: is it necessary to convert the operation to a SingleParticleTransitionMatrix?
        op = _SingleParticleTransitionMatrix(op.single_particle_transition_matrix, op.wires)
        if op.wires in self.op_container:
            new_op = self.op_container[op.wires] @ op
            self.op_container[op.wires] = new_op
        elif len(self) == 0:
            self.op_container[op.wires] = op
        else:
            raise _ContractionMatchgatesContainerAddException(
                f"Operation with wires {op.wires} not in container."
            )
        self.wires_set.update(op.wires.labels)
        return True


class _VHMatchgatesContainer(_ContractionMatchgatesContainer):
    def add(self, op: MatchgateOperation):
        # op = _SingleParticleTransitionMatrix(op.single_particle_transition_matrix, op.wires)
        if op.wires in self.op_container:
            new_op = self.op_container[op.wires] @ op
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


def contract_matchgates(
        operations: Iterable[MatchgateOperation],
        method: Optional[Literal["vertical", "horizontal", "vh", "neighbours"]] = None
) -> Optional[_SingleParticleTransitionMatrix]:
    if method is None:
        return
