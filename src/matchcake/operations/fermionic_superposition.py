from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

from .comp_hh import CompHH
from .fermionic_swap import fSWAP
from .single_particle_transition_matrices.sptm_fermionic_superposition import (
    SptmFermionicSuperposition,
)


class FermionicSuperposition(Operation):
    num_wires = AnyWires
    grad_method = None

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

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

    def __init__(self, wires, id=None, **kwargs):
        r"""
        Construct a new Matchgate Superposition operation.
        After applying this operation on the vacuum state each even modes will be in equal superposition.

        :Note: The number of wires must be even.

        :param wires: The wires to embed the features on.
        :param id: The id of the operation.

        :keyword contract_rots: If True, contract the rotations. Default is False.
        """
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    def to_sptm_operation(self):
        return SptmFermionicSuperposition(wires=self.wires)
