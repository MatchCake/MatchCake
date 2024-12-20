import numpy as np
from ...utils.torch_utils import to_numpy

from .optimizer_strategy import OptimizerStrategy
from typing import Dict, Any, Callable, Optional, List

import torch
from pennylane.typing import TensorLike


class GridSearchStrategy(OptimizerStrategy):
    NAME: str = "GridSearch"
    REQUIRES_HYPERPARAMETERS = []
    OPTIONAL_HYPERPARAMETERS = []

    def __init__(self):
        super().__init__()
        self.temperature = None
        self.current_temperature = self.temperature
        self.learning_rate = None
        self.current_loss = np.inf
        self.current_itr = 0
        self.beta_grid = None
        self.gamma_grid = None

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state.update({
            "current_loss": self.current_loss,
            "parameters": to_numpy(self.params_vector).tolist()
        })
        return state

    def __setstate__(self, state: Dict[str, Any]):
        state = super().__setstate__(state)
        self.current_loss = state["current_loss"]
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
        return self

    def step(
            self,
            closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
            callback: Optional[Callable[[], Any]] = None
    ) -> TensorLike:
        if self.parameters is None:
            raise ValueError(f"{self.NAME} Optimizer has not been initialized. Call set_parameters() first.")

        current_beta_vector = self.beta_vector
        grid_coords = self.current_itr % len(self.beta_grid), self.current_itr // len(self.beta_grid)
        beta_candidate_vector = torch.from_numpy(
            np.array([self.beta_grid[grid_coords[0]], self.gamma_grid[grid_coords[1]]])
        ).to(current_beta_vector.device)


        current_gamma_vector = self.gamma_vector
        gamma_candidate = torch.from_numpy(
            np.random.choice(SptmMaxCutEdgesCost.EQUAL_ALLOWED_ANGLES, size=current_gamma_vector.numel())
        ).to(current_gamma_vector.device)
        gamma_candidate_vector = torch.nn.utils.parameters_to_vector(gamma_candidate)

        vec_dict = self.to_vec_dict(beta_candidate_vector, gamma_candidate_vector)
        candidate_vector = torch.cat([vec_dict[i] for i in range(len(self.parameters))], dim=-1)

        candidate = self.vector_to_parameters(candidate_vector)
        candidate_loss = to_numpy(closure(candidate))
        if not np.isfinite(self.current_loss):
            self.current_loss = candidate_loss
        if candidate_loss < self.current_loss:
            self.parameters = candidate
            self.current_loss = candidate_loss
        else:
            vec_dict = self.to_vec_dict(current_beta_vector, current_gamma_vector)
            current_params_vector = torch.cat([vec_dict[i] for i in range(len(self.parameters))], dim=-1)
            self.parameters = self.vector_to_parameters(current_params_vector)
        if callback is not None:
            callback()
        return candidate_loss

    def make_beta_gamma_grid(self, n: int):
        """
        Make a grid of n points for beta and gamma parameters.
        """
        n_beta = self.beta_vector.numel()
        n_gamma = self.gamma_vector.numel()
        gamma_space_size = len(SptmMaxCutEdgesCost.EQUAL_ALLOWED_ANGLES)
        beta_space_size = int(n / gamma_space_size)

        beta_space = np.linspace(self.init_range_low, self.init_range_high, beta_space_size)
        gamma_space = np.array(SptmMaxCutEdgesCost.EQUAL_ALLOWED_ANGLES)

        coords_list = [
            ()
        ]



        beta_grid, gamma_grid = np.meshgrid(beta_space, gamma_space)
        self.beta_grid = beta_grid.flatten()
        self.gamma_grid = gamma_grid.flatten()

        # create a coords matrix of size (n_beta, n_gamma, beta_space_size, gamma_space_size)

        coords = np.zeros((n_beta, n_gamma, beta_space_size, gamma_space_size))


        return beta_grid, gamma_grid

    def optimize(
            self,
            *,
            n_iterations: int,
            closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
            callback: Optional[Callable[[], Any]] = None,
            **hyperparameters
    ) -> List[torch.nn.Parameter]:
        self.make_beta_gamma_grid(n_iterations)
        for i in range(n_iterations):
            self.current_itr = i
            self.step(closure, callback)
        return self.parameters
