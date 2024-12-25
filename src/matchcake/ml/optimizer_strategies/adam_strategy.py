import numpy as np

from .adamw_strategy import AdamWStrategy
from .optimizer_strategy import OptimizerStrategy
from typing import Dict, Any, Callable, Optional, List

import torch
from pennylane.typing import TensorLike


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
