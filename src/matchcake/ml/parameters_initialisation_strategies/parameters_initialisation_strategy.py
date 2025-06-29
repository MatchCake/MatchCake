from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
from pennylane.typing import TensorLike

from ...utils import torch_utils


class ParametersInitialisationStrategy(ABC):
    NAME: str = "ParametersInitialisationStrategy"
    REQUIRES_HYPERPARAMETERS = []
    OPTIONAL_HYPERPARAMETERS = []

    def check_required_hyperparameters(self, hyperparameters):
        for kwarg in self.REQUIRES_HYPERPARAMETERS:
            if kwarg not in hyperparameters:
                raise ValueError(f"Missing required hyperparameter: {kwarg}")
        return self

    def set_optional_hyperparameters(self, hyperparameters, default=None):
        for kwarg in self.OPTIONAL_HYPERPARAMETERS:
            setattr(self, kwarg, hyperparameters.get(kwarg, getattr(self, kwarg, default)))
        return self

    def __init__(self):
        self.parameters_memory = []
        self.init_range_low = 0.0
        self.init_range_high = 4 * np.pi

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def __setstate__(self, state: Dict[str, Any]):
        return self

    def add_parameters_to_memory(self, parameters):
        self.parameters_memory.append(torch_utils.to_numpy(parameters))
        return self

    @abstractmethod
    def initialise_parameters(self, **hyperparameters):
        raise NotImplementedError(f"{self.NAME}.initialise_parameters() must be implemented.")

    @abstractmethod
    def get_next_parameters(self, step_id: int, **hyperparameters):
        raise NotImplementedError(f"{self.NAME}.get_next_parameters() must be implemented.")
