import numpy as np
from .optimizer_strategy import OptimizerStrategy
from typing import Dict, Any, Callable, Optional, List

import torch
from pennylane.typing import TensorLike

from ...utils.torch_utils import torch_wrap_circular_bounds, to_numpy


class FermiQAOASimulatedAnnealingStrategy(OptimizerStrategy):
    NAME: str = "fQAOASimulatedAnnealing"
    REQUIRES_HYPERPARAMETERS = ["learning_rate", "temperature"]
    OPTIONAL_HYPERPARAMETERS = ["seed"]

    def __init__(self):
        super().__init__()
        self.temperature = None
        self.current_temperature = self.temperature
        self.learning_rate = None
        self.current_loss = np.inf
        self.seed = None
        self.np_rng = np.random.default_rng(self.seed)

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state.update({
            "temperature": self.temperature,
            "current_temperature": self.current_temperature,
            "current_loss": self.current_loss,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "parameters": to_numpy(self.params_vector).tolist()
        })
        return state

    def __setstate__(self, state: Dict[str, Any]):
        state = super().__setstate__(state)
        self.temperature = state["temperature"]
        self.current_temperature = state["current_temperature"]
        self.current_loss = state["current_loss"]
        self.learning_rate = state["learning_rate"]
        self.seed = state["seed"]
        self.np_rng = np.random.default_rng(self.seed)
        self.parameters = self.vector_to_parameters(state["parameters"])
        return self

    @property
    def beta_vector(self):
        beta_idx = self.names.index("beta")
        return torch.nn.utils.parameters_to_vector(self.parameters[beta_idx])

    @property
    def gamma_vector(self):
        gamma_idx = self.names.index("gamma")
        return torch.nn.utils.parameters_to_vector(self.parameters[gamma_idx])

    def to_vec_dict(self, beta, gamma):
        return {
            self.names.index("beta"): beta,
            self.names.index("gamma"): gamma
        }

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        self.learning_rate = hyperparameters["learning_rate"]
        self.temperature = hyperparameters["temperature"]
        self.seed = hyperparameters.get("seed", None)
        self.np_rng = np.random.default_rng(self.seed)
        return self

    def step(
            self,
            closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
            callback: Optional[Callable[[], Any]] = None
    ) -> TensorLike:
        if self.parameters is None:
            raise ValueError(f"{self.NAME} Optimizer has not been initialized. Call set_parameters() first.")

        current_beta_vector = self.beta_vector
        beta_candidate_vector = torch_wrap_circular_bounds(
            current_beta_vector + torch.randn_like(current_beta_vector) * self.learning_rate,
            lower_bound=self.init_range_low, upper_bound=self.init_range_high
        )

        current_gamma_vector = self.gamma_vector
        delta_gamma = torch.from_numpy(
            np.random.choice(np.diff(SptmMaxCutEdgesCost.EQUAL_ALLOWED_ANGLES), size=current_gamma_vector.numel())
        ).to(current_gamma_vector.device)
        rn_sign = torch.from_numpy(
            self.np_rng.choice([-1, 1], size=current_gamma_vector.numel())
        ).to(current_gamma_vector.device)
        gamma_candidate_vector = current_gamma_vector + delta_gamma * rn_sign

        vec_dict = self.to_vec_dict(beta_candidate_vector, gamma_candidate_vector)
        candidate_vector = torch.cat([vec_dict[i] for i in range(len(self.parameters))], dim=-1)
        candidate = self.vector_to_parameters(candidate_vector)

        candidate_loss = to_numpy(closure(candidate))
        if not np.isfinite(self.current_loss):
            self.current_loss = candidate_loss
        diff = candidate_loss - self.current_loss
        metropolis = np.exp(-diff / self.current_temperature)
        rn_number = self.np_rng.random()
        if candidate_loss < self.current_loss or rn_number < metropolis:
            self.parameters = candidate
            self.current_loss = candidate_loss
        else:
            vec_dict = self.to_vec_dict(current_beta_vector, current_gamma_vector)
            current_params_vector = torch.cat([vec_dict[i] for i in range(len(self.parameters))], dim=-1)
            self.parameters = self.vector_to_parameters(current_params_vector)
        if callback is not None:
            callback(postfix=dict(temperature=self.current_temperature, metropolis=metropolis))
        return candidate_loss

    def optimize(
            self,
            *,
            n_iterations: int,
            closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
            callback: Optional[Callable[[], Any]] = None,
            **hyperparameters
    ) -> List[torch.nn.Parameter]:
        self.current_loss = to_numpy(closure(self.parameters))
        for i in range(n_iterations):
            self.current_temperature = self.temperature / (i + 1)
            self.step(closure, callback)
        return self.parameters
