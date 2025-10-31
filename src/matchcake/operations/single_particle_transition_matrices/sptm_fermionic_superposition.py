import numpy as np
from pennylane.wires import Wires

from ...utils import make_wires_continuous
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation
from .sptm_comp_hh import SptmCompHH
from .sptm_fswap import SptmCompZX


class SptmFermionicSuperposition(SingleParticleTransitionMatrixOperation):
    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = Wires(wires)
        gates = []
        for wire_i, wire_j in zip(wires[:-1], wires[1:]):
            new_gates = [
                SptmCompZX(wires=[wire_i, wire_j]),
                SptmCompHH(wires=[wire_i, wire_j]),
            ]
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
