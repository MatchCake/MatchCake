from abc import ABC, abstractmethod
from typing import Sequence

from pennylane.operation import Operation
import tqdm

from ...operations.matchgate_operation import MatchgateOperation
from ...operations.single_particle_transition_matrices import SingleParticleTransitionMatrixOperation
from .contraction_container import _ContractionMatchgatesContainer


class ContractionStrategy(ABC):
    NAME: str = "ContractionStrategy"
    ALLOWED_GATE_CLASSES = [MatchgateOperation, SingleParticleTransitionMatrixOperation]

    def __init__(self, show_progress: bool = False):
        self.p_bar = None
        self.show_progress = show_progress

    @abstractmethod
    def get_container(self) -> _ContractionMatchgatesContainer:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def __call__(
            self,
            operations: Sequence[Operation],
            **kwargs
    ) -> Sequence[Operation]:
        self.p_bar = kwargs.get("p_bar", None)
        self.show_progress = kwargs.get("show_progress", self.show_progress)

        if len(operations) <= 1:
            return operations
        new_operations = []
        container = self.get_container()

        self.initialize_p_bar(total=len(operations), initial=0, desc=f"{self.NAME} contraction")
        container.contract_operations(operations=operations, callback=self.p_bar_set_n_p1)
        # for i, op in enumerate(operations):
        #     if not isinstance(op, tuple(self.ALLOWED_GATE_CLASSES)):
        #         new_operations.append(op)
        #         if container:
        #             new_operations.append(container.contract())
        #             container.clear()
        #         self.p_bar_set_n(i + 1)
        #         continue
        #     new_op = container.push_contract(op)
        #     if new_op is not None:
        #         new_operations.append(new_op)
        #     self.p_bar_set_n(i + 1)
        # if container:
        #     new_operations.append(container.contract())
        #     container.clear()
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
