from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pennylane.typing import TensorLike

from .optimizer_strategy import OptimizerStrategy


class AdamWStrategy(OptimizerStrategy):
    NAME: str = "AdamW"
    REQUIRES_HYPERPARAMETERS = ["learning_rate", "max_grad_norm"]
    REQUIRES_GRAD = True

    def __init__(self):
        super().__init__()
        self.learning_rate = None
        self.optimizer = None
        self.max_grad_norm = None

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            **{key: getattr(self, key) for key in self.REQUIRES_HYPERPARAMETERS},
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(state["optimizer"])
        for key, value in state.items():
            setattr(self, key, value)
        return self

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        self.learning_rate = hyperparameters["learning_rate"]
        self.max_grad_norm = hyperparameters["max_grad_norm"]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.learning_rate)
        return self

    def step(
        self,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
    ) -> TensorLike:
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call set_parameters() first.")
        loss = closure()
        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"Expected closure to return a torch.Tensor, but got {type(loss)}")
        self.optimizer.zero_grad()
        loss.backward()
        if np.isfinite(self.max_grad_norm):
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()
        if callback is not None:
            callback()
        return loss
