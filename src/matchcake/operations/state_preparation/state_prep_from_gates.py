from functools import cached_property
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pennylane as qml
from pennylane import BasisState, X
from pennylane.operation import Operation, StatePrepBase
from pennylane.wires import WiresLike

from matchcake.typing import TensorLike


class StatePrepFromGates(StatePrepBase):
    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Optional[WiresLike] = None,
        **hyperparameters: Any,
    ) -> List[Operation]:
        gate_generator: Callable[[WiresLike], Iterator[Operation]] = hyperparameters["gate_generator"]
        return [op for op in gate_generator(wires)]

    def __init__(
        self,
        gate_generator: Callable[[WiresLike], Iterator[Operation]],
        wires: Optional[WiresLike] = None,
        id: Optional[str] = None,
    ):
        super().__init__(wires=wires, id=id)
        self.gate_generator = gate_generator
        self.hyperparameters["gate_generator"] = gate_generator

    def state_vector(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        raise NotImplementedError("State vector is not defined for StatePrepFromGates")

    def decomposition_generator(self) -> Iterator[Operation]:
        return self.gate_generator(self.wires)

    def to_basis_state(self) -> BasisState:
        state = np.zeros(len(self.wires), dtype=int)
        for op in self.decomposition_generator():
            if isinstance(op, X):
                idx = int(op.wires.toarray().item())
                state[idx] = (state[idx] + 1) % 2
            elif isinstance(op, qml.Identity):
                pass
            else:
                raise ValueError(f"Unsupported operation: {op} for StatePrepFromGates.to_basis_state")
        return BasisState(state, wires=self.wires)

    @cached_property
    def is_basis_state(self):
        try:
            with qml.QueuingManager.stop_recording():
                self.to_basis_state()
            return True
        except ValueError:
            return False
