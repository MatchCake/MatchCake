import numpy as np
from pennylane.wires import Wires

from ...utils import make_wires_continuous
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmFSwapHH(SingleParticleTransitionMatrixOperation):
    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        all_wires = make_wires_continuous(wires)
        n_wires = len(all_wires)
        matrix = np.zeros((2 * n_wires, 2 * n_wires), dtype=int)
        matrix[..., ::2, ::2] = np.eye(n_wires)
        matrix[..., np.arange(1, 2 * n_wires - 2, 4), np.arange(3, 2 * n_wires, 4)] = 1
        matrix[..., np.arange(3, 2 * n_wires, 4), np.arange(1, 2 * n_wires - 2, 4)] = -1
        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self
