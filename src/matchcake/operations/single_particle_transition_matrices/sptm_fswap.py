import numpy as np
import pennylane as qml
from ...utils.math import convert_and_cast_like
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmFSwap(SingleParticleTransitionMatrixOperation):
    def __init__(self, wires=None, id=None, **kwargs):
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[..., 0, 2] = 1.0
        matrix[..., 1, 3] = 1.0
        matrix[..., 2, 0] = 1.0
        matrix[..., 3, 1] = 1.0
        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self

