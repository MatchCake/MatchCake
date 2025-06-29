import warnings
from typing import Callable, Iterable, List, Literal, Optional, Union

from pennylane.wires import Wires

from ...operations.matchgate_operation import MatchgateOperation
from ...operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
    _SingleParticleTransitionMatrix,
)


class _ContractionMatchgatesContainerAddException(Exception):
    pass


class _ContractionMatchgatesContainer:
    ALLOWED_GATE_CLASSES = [MatchgateOperation, SingleParticleTransitionMatrixOperation]

    def __init__(self):
        self.op_container = {}
        self.wires_set = set({})

    @property
    def all_wires(self) -> Wires:
        return Wires(self.wires_set)

    @property
    def all_cs_wires(self):
        return Wires.all_wires([op.cs_wires for op in self.values()])

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.op_container)

    def values(self):
        return self.op_container.values()

    def keys(self):
        return self.op_container.keys()

    def items(self):
        return self.op_container.items()

    def sorted_keys(self):
        return sorted(self.op_container.keys())

    def sorted_values(self):
        return [self.op_container[key] for key in self.sorted_keys()]

    def add(self, op: Union[MatchgateOperation, SingleParticleTransitionMatrixOperation]) -> bool:
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

    def try_add(self, op: Union[MatchgateOperation, SingleParticleTransitionMatrixOperation]) -> bool:
        try:
            return self.add(op)
        except _ContractionMatchgatesContainerAddException:
            return False
        except Exception as e:
            warnings.warn(f"Unexpected exception in try_add: {e}", RuntimeWarning)
            return False

    def extend(
        self,
        ops: Iterable[Union[MatchgateOperation, SingleParticleTransitionMatrixOperation]],
    ) -> None:
        for op in ops:
            self.add(op)

    def try_extend(
        self,
        ops: Iterable[Union[MatchgateOperation, SingleParticleTransitionMatrixOperation]],
    ) -> int:
        for i, op in enumerate(ops):
            if not self.try_add(op):
                return i
        return -1

    def clear(self):
        self.op_container.clear()
        self.wires_set.clear()

    def contract(self) -> Optional[_SingleParticleTransitionMatrix]:
        return SingleParticleTransitionMatrixOperation.from_operations(self.values())

    def contract_and_clear(self) -> Optional[_SingleParticleTransitionMatrix]:
        contracted_op = self.contract()
        self.clear()
        return contracted_op

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
            contracted_op = self.contract_and_clear()
            self.add(op)
            return contracted_op
        return None

    def contract_operations(
        self, operations, callback: Optional[Callable[[int], None]] = None
    ) -> List[Union[MatchgateOperation, _SingleParticleTransitionMatrix]]:
        new_operations = []
        for i, op in enumerate(operations):
            if not isinstance(op, tuple(self.ALLOWED_GATE_CLASSES)):
                if self:
                    new_operations.append(self.contract_and_clear())
                new_operations.append(op)
                if callback is not None:
                    callback(i)
                continue
            new_op = self.push_contract(op)
            if new_op is not None:
                new_operations.append(new_op)
            if callback is not None:
                callback(i)
        if self:
            new_operations.append(self.contract_and_clear())
        return new_operations
