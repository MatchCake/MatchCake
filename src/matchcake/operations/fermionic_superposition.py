from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

from .comp_hh import CompHH
from .fermionic_swap import fSWAP
from .single_particle_transition_matrices.sptm_fermionic_superposition import (
    SptmFermionicSuperposition,
)


class FermionicSuperposition(Operation):
    """
    Represents a Fermionic Superposition operation.

    This class implements a quantum operation where, when applied to the vacuum state,
    each even mode will be in an equal superposition. It allows for operations over
    arbitrary numbers of qubits and is specifically designed to work with even numbers of wires.
    The operation is configured with a set of wires and optional parameters. Additionally,
    it supports features such as decomposition into fundamental gates.
    """

    num_wires = AnyWires
    grad_method = None

    @classmethod
    def random(cls, wires: Wires, **kwargs):
        return cls(wires=wires, id=kwargs.get("id", None))

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = Wires(wires)
        gates = []
        for wire_i, wire_j in zip(wires[:-1], wires[1:]):
            gates.append(fSWAP(wires=[wire_i, wire_j]))
            gates.append(CompHH(wires=[wire_i, wire_j]))
        return gates

    def __repr__(self):
        return f"{self.__class__.__name__}(wires={self.wires.tolist()})"

    @property
    def num_params(self):
        return 0

    def to_sptm_operation(self) -> SptmFermionicSuperposition:
        return SptmFermionicSuperposition(wires=self.wires, id=self.id, **self.hyperparameters)
