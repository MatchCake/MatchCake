from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pennylane.typing import TensorLike

from ...utils import torch_utils
from .optimizer_strategy import OptimizerStrategy


class GeneticStrategy(OptimizerStrategy):
    NAME: str = "Genetic"
    OPTIONAL_HYPERPARAMETERS = [
        "init_range_low",
        "init_range_high",
        "num_parents_mating",
        "sol_per_pop",
        "parent_selection_type",
        "keep_parents",
        "crossover_type",
        "mutation_type",
        "mutation_percent_genes",
    ]

    def __init__(self):
        super().__init__()
        self.num_parents_mating = 4
        self.sol_per_pop = 8
        self.parent_selection_type = "sss"
        self.keep_parents = -1
        self.crossover_type = "single_point"
        self.mutation_type = "random"
        self.mutation_percent_genes = "default"
        self.closure = None
        self.callback = None

    def __getstate__(self) -> Dict[str, Any]:
        return super().__getstate__()

    def __setstate__(self, state: Dict[str, Any]):
        return super().__setstate__(state)

    def set_parameters(self, parameters, **hyperparameters):
        super().set_parameters(parameters, **hyperparameters)
        return self

    def step(
        self,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
    ) -> TensorLike:
        raise NotImplementedError(f"{self.NAME}.step() must be implemented.")

    def on_generation(self, ga_instance):
        if self.callback is not None:
            self.callback()

    def fitness_func(self, ga_instance, solution, solution_idx):
        if self.closure is None:
            raise ValueError(f"{self.NAME} Optimizer has not been initialized. Call optimize() first.")
        return -float(torch_utils.to_numpy(self.closure(self.vector_to_parameters(solution))))

    def get_initial_population(self):
        main_parent = torch_utils.to_numpy(self.params_vector)
        initial_population = [main_parent]
        for _ in range(max(0, self.sol_per_pop - 1)):
            initial_population.append(main_parent + torch_utils.to_numpy(torch.randn_like(self.params_vector)))
        return np.stack(initial_population)

    def optimize(
        self,
        *,
        n_iterations: int,
        closure: Callable[[Optional[List[torch.nn.Parameter]]], TensorLike],
        callback: Optional[Callable[[], Any]] = None,
        **hyperparameters,
    ) -> List[torch.nn.Parameter]:
        try:
            import pygad
        except ImportError:
            raise ImportError(
                "Please install pygad to use the Genetic optimizer strategy. "
                "You can install it with `pip install pygad`."
            )
        if self.parameters is None:
            raise ValueError(f"{self.NAME} Optimizer has not been initialized. Call set_parameters() first.")

        self.closure = closure
        self.callback = callback
        ga_instance = pygad.GA(
            num_generations=n_iterations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            initial_population=self.get_initial_population(),
            sol_per_pop=self.sol_per_pop,
            num_genes=len(self.params_vector),
            init_range_low=self.init_range_low,
            init_range_high=self.init_range_high,
            parent_selection_type=self.parent_selection_type,
            keep_parents=self.keep_parents,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_percent_genes=self.mutation_percent_genes,
            on_generation=self.on_generation,
        )
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        self.parameters = self.vector_to_parameters(solution)
        return self.parameters
