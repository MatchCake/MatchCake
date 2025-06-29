from typing import Callable, Optional

import numpy as np
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
from ...base.lookup_table import NonInteractingFermionicLookupTable
from .probability_strategy import ProbabilityStrategy


class LookupTableStrategy(ProbabilityStrategy):
    NAME: str = "LookupTable"
    REQUIRES_KWARGS = ["lookup_table", "pfaffian_method"]

    def __call__(
        self,
        *,
        system_state: TensorLike,
        target_binary_state: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        self.check_required_kwargs(kwargs)
        if isinstance(wires, int):
            wires = [wires]
        wires = Wires(wires)
        all_wires = kwargs.get("all_wires", wires)
        wires_indexes = all_wires.indices(wires)

        lookup_table: NonInteractingFermionicLookupTable = kwargs["lookup_table"]
        pfaffian_method: str = kwargs["pfaffian_method"]

        show_progress = kwargs.get("show_progress", False)
        obs = lookup_table.get_observable_of_target_state(
            system_state,
            target_binary_state,
            wires_indexes,
            show_progress=show_progress,
        )
        prob = qml.math.real(utils.pfaffian(obs, method=pfaffian_method, show_progress=show_progress))
        return prob

    def batch_call(
        self,
        *,
        system_state: TensorLike,
        target_binary_states: TensorLike,
        batch_wires: Optional[Wires] = None,
        **kwargs,
    ) -> TensorLike:
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
