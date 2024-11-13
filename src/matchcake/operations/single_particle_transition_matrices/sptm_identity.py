import numpy as np
from pennylane.wires import Wires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmIdentity(SingleParticleTransitionMatrixOperation):
    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        matrix = np.eye(4, dtype=complex)
        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self
