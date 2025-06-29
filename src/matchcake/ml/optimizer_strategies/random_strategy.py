from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pennylane.typing import TensorLike

from .optimizer_strategy import OptimizerStrategy


class RandomStrategy(OptimizerStrategy):
    NAME: str = "Random"
    REQUIRES_HYPERPARAMETERS = []

    def __init__(self):
        super().__init__()

    def __getstate__(self) -> Dict[str, Any]:
        return super().__getstate__()

    def __setstate__(self, state: Dict[str, Any]):
        return super().__setstate__(state)

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        return self

    def step(
        self,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
    ) -> TensorLike:
        if self.parameters is None:
            raise ValueError(f"{self.NAME} Optimizer has not been initialized. Call set_parameters() first.")

        vec = torch.nn.utils.parameters_to_vector(self.parameters)
        vec += torch.randn_like(vec)
        self.parameters = self.vector_to_parameters(vec)
        loss = closure(self.parameters)
        if callback is not None:
            callback()
        return loss
