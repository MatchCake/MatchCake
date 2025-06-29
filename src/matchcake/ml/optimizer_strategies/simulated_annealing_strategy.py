from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pennylane.typing import TensorLike

from ...utils.torch_utils import to_numpy, torch_wrap_circular_bounds
from .optimizer_strategy import OptimizerStrategy


class SimulatedAnnealingStrategy(OptimizerStrategy):
    NAME: str = "SimulatedAnnealing"
    REQUIRES_HYPERPARAMETERS = ["learning_rate"]
    OPTIONAL_HYPERPARAMETERS = ["seed", "temperature"]
    DEFAULT_TEMPERATURE = 10.0

    def __init__(self):
        super().__init__()
        self.temperature = self.DEFAULT_TEMPERATURE
        self.current_temperature = self.temperature
        self.learning_rate = None
        self.current_loss = np.inf
        self.seed = None
        self.np_rng = np.random.default_rng(self.seed)

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state.update(
            {
                "temperature": self.temperature,
                "current_temperature": self.current_temperature,
                "current_loss": self.current_loss,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "parameters": to_numpy(self.params_vector).tolist(),
            }
        )
        return state

    def __setstate__(self, state: Dict[str, Any]):
        state = super().__setstate__(state)
        self.temperature = state.get("temperature", self.temperature)
        self.current_temperature = state["current_temperature"]
        self.current_loss = state["current_loss"]
        self.learning_rate = state["learning_rate"]
        self.seed = state["seed"]
        self.np_rng = np.random.default_rng(self.seed)
        self.parameters = self.vector_to_parameters(state["parameters"])
        return self

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        self.learning_rate = hyperparameters["learning_rate"]
        self.temperature = hyperparameters.get("temperature", self.temperature)
        self.seed = hyperparameters.get("seed", None)
        self.np_rng = np.random.default_rng(self.seed)
        return self

    def step(
        self,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
    ) -> TensorLike:
        if self.parameters is None:
            raise ValueError(f"{self.NAME} Optimizer has not been initialized. Call set_parameters() first.")

        current_params_vector = deepcopy(self.params_vector)
        # candidate_vector = current_params_vector + torch.randn_like(current_params_vector) * self.learning_rate
        # candidate_vector = torch.clamp(candidate_vector, self.init_range_low, self.init_range_high)
        candidate_vector = torch_wrap_circular_bounds(
            current_params_vector + torch.randn_like(current_params_vector) * self.learning_rate,
            lower_bound=self.init_range_low,
            upper_bound=self.init_range_high,
        )

        candidate = deepcopy(self.vector_to_parameters(candidate_vector))
        candidate_loss = to_numpy(closure(candidate))
        if not np.isfinite(self.current_loss):
            self.current_loss = candidate_loss
        diff = candidate_loss - self.current_loss
        metropolis = np.exp(-diff / self.current_temperature)
        rn_number = self.np_rng.random()
        if candidate_loss < self.current_loss or rn_number < metropolis:
            self.parameters = deepcopy(candidate)
            self.current_loss = candidate_loss
        else:
            self.parameters = deepcopy(self.vector_to_parameters(current_params_vector))
        if candidate_loss < self.best_cost:
            self.best_parameters = deepcopy(self.parameters)
            self.best_cost = candidate_loss
        if callback is not None:
            callback(postfix=dict(temperature=self.current_temperature, metropolis=metropolis))
        return candidate_loss

    def optimize(
        self,
        *,
        n_iterations: int,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
        **hyperparameters,
    ) -> List[torch.nn.Parameter]:
        for i in range(n_iterations):
            self.current_temperature = self.temperature / (i + 1)
            self.step(closure, callback)
            if self.stop_training_flag:
                break
        return self.best_parameters
