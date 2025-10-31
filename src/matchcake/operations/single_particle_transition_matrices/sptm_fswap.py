import numpy as np
from pennylane.wires import Wires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmCompZX(SingleParticleTransitionMatrixOperation):
    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        wires_arr = Wires(wires).toarray()
        wire0, wire1 = np.sort(wires_arr)
        size = 2 * (wire1 - wire0 + 1)
        matrix = np.zeros((size, size), dtype=int)

        first_idx = 2 * (wire1 - wire0)
        matrix[:2, first_idx : first_idx + 2] = np.eye(2)

        rows, cols = np.where(np.eye(*matrix.shape, k=-2))
        matrix[rows, cols] = 1

        if wire0 != wires_arr[0]:
            matrix = matrix.T

        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self


SptmFSwap = SptmCompZX
SptmFSwap.__name__ = "SptmFSwap"
