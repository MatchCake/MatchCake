from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, cast

import tqdm
from pennylane.operation import Operation
from pennylane.typing import TensorLike

from ...operations.matchgate_operation import MatchgateOperation
from ...operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from ..sptm_circuit_records import SptmCircuitRecords
from .contraction_container import _ContractionMatchgatesContainer


class ContractionStrategy(ABC):
    NAME: str = "ContractionStrategy"
    ALLOWED_GATE_CLASSES = [MatchgateOperation, SingleParticleTransitionMatrixOperation]

    def __init__(self, show_progress: bool = False):
        self.p_bar = None
        self.show_progress = show_progress
        self.container = self.get_container()
        self._dtype_name: Optional[str] = None

    @abstractmethod
    def get_container(self) -> _ContractionMatchgatesContainer:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def get_next_operations(self, operation) -> Sequence[Optional[Operation]]:
        if not isinstance(operation, tuple(self.container.ALLOWED_GATE_CLASSES)):
            return [self.container.contract_and_clear(), operation]
        return [self.container.push_contract(cast(MatchgateOperation, operation))]

    def get_reminding(self):
        return self.container.contract_and_clear()

    def build_plan(
        self, records: SptmCircuitRecords, n_columns: int, dtype_name: str
    ) -> Optional[List[Tuple[Any, TensorLike]]]:
        """Compile the operations recorded by the device into stacks of blocks to multiply its matrix by.

        The device calls this when its global single-particle transition matrix is read, with the
        operations this strategy gave it since the last read. A strategy that has nothing to gain
        from seeing them together returns None, and the device multiplies them one by one.

        :param records: The operations applied to the device since its global matrix was last read.
        :type records: SptmCircuitRecords
        :param n_columns: The number of columns of the global matrix.
        :type n_columns: int
        :param dtype_name: The name of the real dtype of the device, such as ``"float64"``.
        :type dtype_name: str
        :return: For each step, the columns each block acts on, shape (n_blocks, 2k), and the blocks,
            shape (n_blocks, 2k, 2k) or (batch, n_blocks, 2k, 2k), or None.
        :rtype: Optional[List[Tuple[Any, TensorLike]]]
        """
        return None

    def reset(self):
        if self.container is not None:
            self.container.clear()
        return self

    def __call__(self, operations: Sequence[Operation], **kwargs) -> Sequence[Operation]:
        self.p_bar = kwargs.get("p_bar", None)
        self.show_progress = kwargs.get("show_progress", self.show_progress)

        if len(operations) <= 1:
            return operations

        self.initialize_p_bar(total=len(operations), initial=0, desc=f"{self.NAME} contraction")
        new_operations = self.container.contract_operations(operations=operations, callback=self.p_bar_set_n_p1)
        self.p_bar_set_n(len(operations))
        self.close_p_bar()
        return new_operations

    def p_bar_set_n(self, n: int):
        if self.p_bar is not None:
            self.p_bar.n = n
            self.p_bar.refresh()

    def p_bar_set_n_p1(self, n: int):
        return self.p_bar_set_n(n + 1)

    def initialize_p_bar(self, *args, **kwargs):
        kwargs.setdefault("disable", not self.show_progress)
        if self.p_bar is None and not self.show_progress:
            return
        self.p_bar = tqdm.tqdm(*args, **kwargs)
        return self.p_bar

    def close_p_bar(self):
        if self.p_bar is not None:
            self.p_bar.close()

    @property
    def dtype_name(self) -> Optional[str]:
        """The name of the real dtype the operations are contracted in, None to keep that of each operation.

        The device sets it to its own precision, so that a strategy multiplies in that precision.

        :return: The dtype name, such as ``"float32"``.
        :rtype: Optional[str]
        """
        return self._dtype_name

    @dtype_name.setter
    def dtype_name(self, value: Optional[str]) -> None:
        self._dtype_name = value
        if self.container is not None:
            self.container.dtype_name = value
