import numpy as np
from pennylane.wires import Wires

from ...constants import _CIRCUIT_MATMUL_DIRECTION
from ...utils.math import dagger
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmCompHH(SingleParticleTransitionMatrixOperation):
    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[..., 0, 2] = 1.0
        matrix[..., 1, 1] = -1.0
        matrix[..., 2, 0] = 1.0
        matrix[..., 3, 3] = 1.0
        # if _MATMUL_DIRECTION == "rl":
        #     matrix = dagger(matrix)
        # else:
        #     matrix = dagger(matrix)
        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self
