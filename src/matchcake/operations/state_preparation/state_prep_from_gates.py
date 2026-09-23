from functools import cached_property
from typing import Any, Callable, Iterator, List, Optional

import numpy as np
import pennylane as qml
from pennylane import BasisState, X
from pennylane.operation import Operation
from pennylane.wires import Wires, WiresLike

from matchcake.typing import TensorLike

from .product_state import ProductState


class StatePrepFromGates(ProductState):
    r"""Prepare a state by listing the gates that build it from :math:`|0\ldots0\rangle`.

    This is a :class:`ProductState` whose per-qubit amplitudes are obtained by
    applying the gates yielded by ``gate_generator`` to the all-zero state. Only
    *single-qubit* gate generators (one or more gates acting on individual wires)
    describe a product state and are therefore supported; a multi-qubit gate would
    entangle the qubits and cannot be represented as a :class:`ProductState`.

    Because it is a genuine :class:`ProductState`, it works with every strategy that
    accepts product-state inputs (e.g. the m-Pfaffian expectation-value strategy),
    while still decomposing into its defining gates for the gate-based paths.
    """

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Optional[WiresLike] = None,
        **hyperparameters: Any,
    ) -> List[Operation]:
        gate_generator: Callable[[WiresLike], Iterator[Operation]] = hyperparameters["gate_generator"]
        return [op for op in gate_generator(wires)]

    @staticmethod
    def _product_amplitudes_from_gates(
        gate_generator: Callable[[WiresLike], Iterator[Operation]],
        wires: WiresLike,
    ) -> np.ndarray:
        """Compute the ``(n, 2)`` per-qubit amplitudes produced by the gate generator.

        Each qubit starts in :math:`|0\\rangle` and the single-qubit gates yielded by
        ``gate_generator`` are applied in order. A gate acting on more than one wire
        entangles the qubits and is rejected with a :class:`ValueError`.
        """
        wires = Wires(wires)
        n = len(wires)
        wire_to_index = {w: i for i, w in enumerate(wires)}

        # Each qubit starts in |0> = [1, 0].
        amplitudes = np.zeros((n, 2), dtype=complex)
        amplitudes[:, 0] = 1.0

        # Stop recording so that materialising the gates here does not queue them onto
        # the active tape (the StatePrepFromGates op itself is what gets queued).
        with qml.QueuingManager.stop_recording():
            for op in gate_generator(wires):
                if isinstance(op, qml.Identity):
                    continue
                if len(op.wires) != 1:
                    raise ValueError(
                        f"StatePrepFromGates can only be represented as a ProductState when its "
                        f"gate generator yields single-qubit gates, but got {op} acting on "
                        f"{len(op.wires)} wires."
                    )
                idx = wire_to_index[op.wires[0]]
                matrix = np.asarray(qml.math.toarray(op.matrix()), dtype=complex)
                amplitudes[idx] = matrix @ amplitudes[idx]
        return amplitudes

    def __init__(
        self,
        gate_generator: Callable[[WiresLike], Iterator[Operation]],
        wires: Optional[WiresLike] = None,
    ):
        self.gate_generator = gate_generator
        amplitudes = self._product_amplitudes_from_gates(gate_generator, wires)
        super().__init__(amplitudes, wires=wires, validate_norm=False)
        self.hyperparameters["gate_generator"] = gate_generator

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
