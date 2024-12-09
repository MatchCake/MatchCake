from typing import Sequence, Optional

from .contraction_strategy import ContractionStrategy
from pennylane.operation import Operation

from ...operations import SingleParticleTransitionMatrixOperation


class NoneContractionStrategy(ContractionStrategy):
    NAME: str = "None"

    def get_container(self):
        return None

    def __call__(
            self,
            operations: Sequence[Operation],
            **kwargs
    ) -> Sequence[Operation]:
        return operations

    def get_next_operations(self, operation) -> Sequence[Optional[Operation]]:
        return [SingleParticleTransitionMatrixOperation.from_operation(operation)]

    def get_reminding(self):
        return None
