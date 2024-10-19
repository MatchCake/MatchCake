from .probability_strategy import ProbabilityStrategy
from typing import Callable
import numpy as np
from pennylane.typing import TensorLike
from pennylane.wires import Wires
import pennylane as qml

from ...base.lookup_table import NonInteractingFermionicLookupTable
from ... import utils


class LookupTableStrategyV2(ProbabilityStrategy):
    NAME: str = "LookupTableV2"
    REQUIRES_KWARGS = ["lookup_table"]

    def __call__(
            self,
            *,
            system_state: TensorLike,
            target_binary_state: TensorLike,
            wires: Wires,
            **kwargs
    ) -> TensorLike:
        self.check_required_kwargs(kwargs)
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        all_wires = kwargs.get("all_wires", wires)
        wires_indexes = all_wires.indices(wires)

        lookup_table: NonInteractingFermionicLookupTable = kwargs["lookup_table"]
        show_progress = kwargs.get("show_progress", False)
        prob = lookup_table.compute_pfaffian_of_target_state(
            system_state,
            target_binary_state,
            wires_indexes,
            show_progress=show_progress,
        )
        return prob

    def batch_call(
            self,
            *,
            system_state: TensorLike,
            target_binary_states: TensorLike,
            batch_wires: Wires,
            **kwargs
    ) -> TensorLike:
        return super().batch_call(
            system_state=system_state,
            target_binary_states=target_binary_states,
            batch_wires=batch_wires,
            **kwargs
        )
        self.check_required_kwargs(kwargs)

        lookup_table: NonInteractingFermionicLookupTable = kwargs["lookup_table"]
        pfaffian_method: str = kwargs["pfaffian_method"]

        show_progress = kwargs.get("show_progress", False)
        batch_obs = lookup_table.compute_observables_of_target_states(
            system_state,
            target_binary_states,
            batch_wires,
            show_progress=show_progress,
        )
        prob = qml.math.real(utils.pfaffian(batch_obs, method=pfaffian_method, show_progress=show_progress))
        return prob


