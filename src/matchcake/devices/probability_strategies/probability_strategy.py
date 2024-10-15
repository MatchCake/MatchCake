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

    def batch_call(
            self,
            *,
            system_state: TensorLike,
            target_binary_states: TensorLike,
            batch_wires: Wires,
            **kwargs
    ) -> TensorLike:
        if qml.math.shape(target_binary_states) != qml.math.shape(batch_wires):
            raise ValueError(
                f"target_binary_states shape {qml.math.shape(target_binary_states)} does not match "
                f"batch_wires shape {qml.math.shape(batch_wires)}"
            )

        return qml.math.stack([
            self(
                system_state=system_state,
                target_binary_state=target_binary_state,
                wires=wires,
                **kwargs
            )
            for target_binary_state, wires in zip(target_binary_states, batch_wires)
        ])
