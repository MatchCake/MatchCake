from abc import ABC, abstractmethod
from typing import List

import pennylane as qml
from pennylane.operation import StatePrepBase
from pennylane.ops.qubit.state_preparation import BasisState
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pythonbasictools.multiprocessing_tools import apply_func_multiprocess

from matchcake.operations.state_preparation import StatePrepFromGates


class ProbabilityStrategy(ABC):
    NAME: str = "ProbabilityStrategy"
    REQUIRES_KWARGS: List[str] = []

    @classmethod
    def system_basis_state_from_state_prep_op(cls, state_prep_op: StatePrepBase):
        if isinstance(state_prep_op, BasisState):
            return state_prep_op.parameters[0]
        elif isinstance(state_prep_op, StatePrepFromGates):
            return state_prep_op.to_basis_state().parameters[0]
        else:
            raise NotImplementedError(f"{cls.__name__} cannot be used with {type(state_prep_op)}")

    def check_required_kwargs(self, kwargs):
        for kwarg in self.REQUIRES_KWARGS:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
        return self

    @abstractmethod
    def __call__(
            self,
            *,
            # system_state: TensorLike,
            state_prep_op: StatePrepBase,
            target_binary_state: TensorLike,
            wires: Wires,
            **kwargs,
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def batch_call(
            self,
            *,
            # system_state: TensorLike,
            state_prep_op: StatePrepBase,
            target_binary_states: TensorLike,
            batch_wires: Wires,
            **kwargs,
    ) -> TensorLike:
        if qml.math.shape(target_binary_states) != qml.math.shape(batch_wires):
            raise ValueError(
                f"target_binary_states shape {qml.math.shape(target_binary_states)} does not match "
                f"batch_wires shape {qml.math.shape(batch_wires)}"
            )
        show_progress = kwargs.pop("show_progress", False)
        nb_workers = kwargs.pop("nb_workers", 0)
        probs_list = apply_func_multiprocess(
            func=self,
            iterable_of_args=[() for _ in range(len(target_binary_states))],
            iterable_of_kwargs=[
                {
                    # "system_state": system_state,
                    "state_prep_op": state_prep_op,
                    "target_binary_state": target_binary_state,
                    "wires": wires,
                    **kwargs,
                }
                for target_binary_state, wires in zip(target_binary_states, batch_wires)
            ],
            verbose=show_progress,
            desc=f"[{self.NAME}] Batch Probability Calculation with {nb_workers} workers",
            unit="states",
            nb_workers=nb_workers,
        )
        probs = qml.math.stack(probs_list, axis=0)
        return probs
