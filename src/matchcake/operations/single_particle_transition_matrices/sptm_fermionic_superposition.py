import numpy as np
from pennylane.operation import Operation, AnyWires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation
from pennylane.wires import Wires

from . import SptmFSwap, SptmFHH
from ...utils import make_wires_continuous
from ...constants import _CIRCUIT_MATMUL_DIRECTION
from ...utils.math import dagger


class SptmFermionicSuperposition(SingleParticleTransitionMatrixOperation):
    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = Wires(wires)
        gates = []
        for wire_i, wire_j in zip(wires[:-1], wires[1:]):
            new_gates = [SptmFSwap(wires=[wire_i, wire_j]), SptmFHH(wires=[wire_i, wire_j])]
            gates.extend(new_gates)
        return gates

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        all_wires = make_wires_continuous(wires)
        n_wires = len(all_wires)
        matrix = np.zeros((2 * n_wires, 2 * n_wires), dtype=int)
        matrix[..., ::2, ::2] = np.eye(n_wires)
        matrix[..., np.arange(3, 2 * n_wires, 2), np.arange(1, 2 * n_wires - 2, 2)] = -1
        matrix[..., 1, -1] = 1
        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self


# class SptmFermionicSuperposition(Operation):
#     num_wires = AnyWires
#     grad_method = None
#
#     @staticmethod
#     def compute_decomposition(*params, wires=None, **hyperparameters):
#         wires = Wires(wires)
#         gates = []
#         for wire_i, wire_j in zip(wires[:-1], wires[1:]):
#             gates.append(SptmFSwap(wires=[wire_i, wire_j]))
#             gates.append(SptmFHH(wires=[wire_i, wire_j]))
#         return gates
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(wires={self.wires.tolist()})"
#
#     def __init__(self, wires, id=None, **kwargs):
#         r"""
#         Construct a new Matchgate Superposition operation.
#         After applying this operation on the vacuum state each even modes will be in equal superposition.
#
#         :Note: The number of wires must be even.
#
#         :param wires: The wires to embed the features on.
#         :param id: The id of the operation.
#
#         :keyword contract_rots: If True, contract the rotations. Default is False.
#         """
#         super().__init__(wires=wires, id=id)
#
#     @property
#     def num_params(self):
#         return 0
