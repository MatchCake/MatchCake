from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pennylane.typing import TensorLike
from scipy.optimize import minimize

from ...utils import torch_utils


class OptimizerStrategy(ABC):
    NAME: str = "OptimizerStrategy"
    REQUIRES_HYPERPARAMETERS = []
    OPTIONAL_HYPERPARAMETERS = []
    REQUIRES_GRAD = False

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
        self.parameters = None
        self.names = None
        self.hyperparameters = None
        self.init_range_low = 0.0
        self.init_range_high = 4 * np.pi
        self.best_parameters = None
        self.best_cost = np.inf
        self.stop_training_flag = False

    def __str__(self):
        return f"{self.NAME}"

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def __setstate__(self, state: Dict[str, Any]):
        return self

    @property
    def params_vector(self):
        return torch.nn.utils.parameters_to_vector(self.parameters)

    def jac(self, vector, closure):
        vector = torch_utils.to_tensor(vector)
        self.parameters = self.vector_to_parameters(vector)
        return torch_utils.to_numpy(torch.autograd.grad(closure(self.parameters), self.parameters)[0])

    def vector_to_parameters(self, vector):
        vector = torch_utils.to_tensor(vector)
        torch.nn.utils.vector_to_parameters(vector, self.parameters)
        return self.parameters

    def set_parameters(self, parameters, **hyperparameters):
        parameters_list, names = [], []
        for i, param in enumerate(parameters):
            if isinstance(param, torch.nn.Parameter):
                parameters_list.append(param)
                names.append(f"param_{i}")
            elif isinstance(param, tuple):
                parameters_list.append(param[1])
                names.append(param[0])
            else:
                raise ValueError(f"Invalid parameter type: {type(param)}")
        self.parameters = parameters_list
        self.names = names
        self.check_required_hyperparameters(hyperparameters)
        self.set_optional_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters
        return self

    @abstractmethod
    def step(
        self,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
    ) -> TensorLike:
        raise NotImplementedError(f"{self.NAME}.step() must be implemented.")

    def optimize(
        self,
        *,
        n_iterations: int,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
        **hyperparameters,
    ) -> List[torch.nn.Parameter]:
        for _ in range(n_iterations):
            self.step(closure, callback)
            if self.stop_training_flag:
                break
        return self.parameters


class ScipyOptimizerStrategy(OptimizerStrategy):
    NAME: str = "ScipyOptimizerStrategy"
    REQUIRES_HYPERPARAMETERS = []

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        return self

    def step(
        self,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
    ) -> TensorLike:
        raise NotImplementedError(f"{self.NAME}.step() must be implemented.")

    def get_callback_func(self, base_callback):
        def callback(*args, **kwargs):
            if base_callback is not None:
                base_callback(*args, **kwargs)
            if self.stop_training_flag:
                # raise StopIteration
                return True

        return callback

    def optimize(
        self,
        *,
        n_iterations: int,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
        **hyperparameters,
    ) -> List[torch.nn.Parameter]:
        result = minimize(
            fun=lambda x: float(torch_utils.to_numpy(closure(self.vector_to_parameters(x)))),
            x0=torch_utils.to_numpy(self.params_vector),
            method=self.NAME,
            callback=self.get_callback_func(callback),
            options={"maxiter": n_iterations},
            bounds=[(self.init_range_low, self.init_range_high)] * len(self.params_vector),
        )
        self.parameters = self.vector_to_parameters(result.x)
        return self.parameters
