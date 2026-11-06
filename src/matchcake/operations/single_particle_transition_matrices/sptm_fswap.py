import numpy as np
from pennylane.wires import Wires

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmCompZX(SingleParticleTransitionMatrixOperation):
    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, **kwargs):
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

        super().__init__(matrix, wires=wires, **kwargs)

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        """Return the inverse of this fermionic swap.

        On adjacent wires the SPTM is a symmetric permutation, so the gate is its own inverse. With a
        gap between the wires the SPTM is a cyclic permutation of the modes in between, and its
        inverse is the transpose.

        :return: The adjoint operation.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        wire0, wire1 = np.sort(Wires(self.wires).toarray())
        if wire1 - wire0 == 1:
            return self
        return super().adjoint()

    def to_matchgate(self):
        from ..fermionic_swap import CompZX

        return CompZX(wires=self.wires)


SptmFSwap = SptmCompZX
SptmFSwap.__name__ = "SptmFSwap"
