from matchcake.operations import fSWAP, fH
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires


class FermionicSuperposition(Operation):
    num_wires = AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = Wires(wires)
        gates = []
        for wire_i, wire_j in zip(wires[:-1], wires[1:]):
            gates.append(fSWAP(wires=[wire_i, wire_j]))
            gates.append(fH(wires=[wire_i, wire_j]))
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
        n_wires = wires if isinstance(wires, int) else len(wires)
        if n_wires % 2 != 0:
            raise ValueError(f"The number of wires must be even but got {n_wires}.")
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0