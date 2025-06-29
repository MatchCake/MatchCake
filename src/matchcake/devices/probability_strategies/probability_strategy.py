from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pythonbasictools.multiprocessing_tools import apply_func_multiprocess


class ProbabilityStrategy(ABC):
    NAME: str = "ProbabilityStrategy"
    REQUIRES_KWARGS: List[str] = []

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
        **kwargs,
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def batch_call(
        self,
        *,
        system_state: TensorLike,
        target_binary_states: TensorLike,
        batch_wires: Wires,
        **kwargs,
    ) -> TensorLike:
        if qml.math.shape(target_binary_states) != qml.math.shape(batch_wires):
            raise ValueError(
                f"target_binary_states shape {qml.math.shape(target_binary_states)} does not match "
                f"batch_wires shape {qml.math.shape(batch_wires)}"
            )
        show_progress = kwargs.pop("show_progress", False)
        nb_workers = kwargs.pop("nb_workers", 0)
        probs_list = apply_func_multiprocess(
            func=self,
            iterable_of_args=[() for _ in range(len(target_binary_states))],
            iterable_of_kwargs=[
                {
                    "system_state": system_state,
                    "target_binary_state": target_binary_state,
                    "wires": wires,
                    **kwargs,
                }
                for target_binary_state, wires in zip(target_binary_states, batch_wires)
            ],
            verbose=show_progress,
            desc=f"[{self.NAME}] Batch Probability Calculation with {nb_workers} workers",
            unit="states",
            nb_workers=nb_workers,
        )
        probs = qml.math.stack(probs_list, axis=0)
        return probs
