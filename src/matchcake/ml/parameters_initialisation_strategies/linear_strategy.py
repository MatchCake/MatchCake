from typing import Any, Callable, Dict

import numpy as np
import torch
from pennylane.typing import TensorLike

from ...utils import torch_utils
from .parameters_initialisation_strategy import ParametersInitialisationStrategy


class LinearStrategy(ParametersInitialisationStrategy):
    NAME: str = "Linear"
    REQUIRES_HYPERPARAMETERS = [
        "n_layers",
        "seed",
        "parameters_rng",
    ]
    OPTIONAL_HYPERPARAMETERS = [
        "noise",
    ]

    def __init__(self):
        super().__init__()
        self.noise = 0.1

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

        n_layers = hyperparameters["n_layers"]
        parameters_rng = hyperparameters["parameters_rng"]

        np_beta = np.linspace(self.init_range_high, self.init_range_low, n_layers)
        np_gamma = np.linspace(self.init_range_low, self.init_range_high, n_layers)

        np_beta += self.noise * parameters_rng.normal(n_layers)
        np_gamma += self.noise * parameters_rng.normal(n_layers)

        beta = torch.nn.Parameter(torch.from_numpy(np_beta).float())
        gamma = torch.nn.Parameter(torch.from_numpy(np_gamma).float())
        return beta, gamma

    def get_next_parameters(self, step_id: int, **hyperparameters):
        parameters_rng = np.random.default_rng(seed=hyperparameters["seed"] + step_id)
        beta, gamma = self.initialise_parameters(parameters_rng=parameters_rng, **hyperparameters)
        return beta, gamma
