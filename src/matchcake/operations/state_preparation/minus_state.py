from typing import Optional

from pennylane import Hadamard, X
from pennylane.wires import WiresLike

from .state_prep_from_gates import StatePrepFromGates


class MinusState(StatePrepFromGates):
    @classmethod
    def gate_generator(cls, wires: WiresLike):
        for wire in wires:
            yield X(wire)
            yield Hadamard(wire)
        return

    def __init__(
            self,
            wires: Optional[WiresLike] = None,
            id: Optional[str] = None,
    ):
        super().__init__(gate_generator=self.gate_generator, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return "|-⟩"
