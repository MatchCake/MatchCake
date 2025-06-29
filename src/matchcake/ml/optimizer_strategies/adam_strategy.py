from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pennylane.typing import TensorLike

from .adamw_strategy import AdamWStrategy
from .optimizer_strategy import OptimizerStrategy


class AdamStrategy(AdamWStrategy):
    NAME: str = "Adam"
    REQUIRES_GRAD = True

    def __init__(self):
        super().__init__()
        self.learning_rate = None
        self.optimizer = None
        self.max_grad_norm = None

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        return self
