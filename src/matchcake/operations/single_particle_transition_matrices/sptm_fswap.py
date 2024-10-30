import numpy as np
from pennylane.wires import Wires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmFSwap(SingleParticleTransitionMatrixOperation):
    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        chain_sptm = kwargs.get("chain_sptm", None)  # TODO: remove, for debugging only
        wires_arr = Wires(wires).toarray()
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[..., 0, 2] = 1.0
        matrix[..., 1, 3] = 1.0
        matrix[..., 2, 0] = 1.0
        matrix[..., 3, 1] = 1.0
        super().__init__(matrix, wires=wires, id=id, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return self

