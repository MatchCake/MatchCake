from typing import Any, Callable, Dict

import numpy as np
import torch
from pennylane.typing import TensorLike

from ...utils import torch_utils
from .parameters_initialisation_strategy import ParametersInitialisationStrategy


class RandomStrategy(ParametersInitialisationStrategy):
    NAME: str = "Random"
    REQUIRES_HYPERPARAMETERS = [
        "current_named_parameters",
        "parameters_rng",
        "seed",
    ]

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def __setstate__(self, state: Dict[str, Any]):
        return self

    def add_parameters_to_memory(self, parameters):
        self.parameters_memory.append(torch_utils.to_numpy(parameters))
        return self

    def initialise_parameters(self, **hyperparameters):
        self.check_required_hyperparameters(hyperparameters)
        self.set_optional_hyperparameters(hyperparameters)

        current_named_parameters = hyperparameters["current_named_parameters"]
        parameters_rng = hyperparameters["parameters_rng"]

        next_named_parameters = []
        for name, parameter in current_named_parameters:
            new_value = parameters_rng.random(size=parameter.shape)
            next_named_parameters.append((name, torch.from_numpy(new_value).float()))
        return next_named_parameters

    def get_next_parameters(self, step_id: int, **hyperparameters):
        parameters_rng = np.random.default_rng(seed=hyperparameters["seed"] + step_id)
        self.initialise_parameters(parameters_rng=parameters_rng, **hyperparameters)
        return self.initialise_parameters(parameters_rng=parameters_rng, **hyperparameters)
