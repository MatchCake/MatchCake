from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from pennylane.typing import TensorLike
from pennylane.wires import Wires
import pennylane as qml


class ProbabilityStrategy(ABC):
    NAME: str = "ProbabilityStrategy"
    REQUIRES_KWARGS = []

    def check_required_kwargs(self, kwargs):
        for kwarg in self.REQUIRES_KWARGS:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
        return self

    @abstractmethod
    def __call__(
            self,
            *,
            system_state: TensorLike,
            target_binary_state: TensorLike,
            wires: Wires,
            **kwargs
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")
