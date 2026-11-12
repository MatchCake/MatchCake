from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pennylane.wires import Wires, WiresLike

from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation

if TYPE_CHECKING:
    from ..fermionic_swap import CompZX


class SptmCompZX(SingleParticleTransitionMatrixOperation):
    r"""
    Single-particle transition matrix (SPTM) of the fermionic swap :class:`~matchcake.operations.CompZX`,
    also called fSWAP, the SWAP gate with a :math:`-1` phase on :math:`|11\rangle`.

    On two adjacent wires :math:`(k, k+1)`, the gate swaps the Majorana pairs of the two modes,
    :math:`(c_{2k}, c_{2k+1}) \leftrightarrow (c_{2k+2}, c_{2k+3})`, so its SPTM is

    .. math::
        R = \begin{bmatrix}
            0 & I_2 \\
            I_2 & 0
        \end{bmatrix}.

    The two wires do not have to be adjacent. For wires ``[wire0, wire1]`` with ``wire0 < wire1``, the
    operation is the chain of adjacent fSWAPs on ``(wire0, wire0 + 1)``, ``(wire0 + 1, wire0 + 2)``,
    ..., ``(wire1 - 1, wire1)``, applied in this order. It moves the mode of ``wire0`` to ``wire1`` and
    shifts every mode in between down by one wire, with the fermionic signs of the swaps included. The
    SPTM is then a cyclic permutation of the Majorana pairs of the modes in that range, of size
    :math:`2(\text{wire1} - \text{wire0} + 1)`. With the wires in decreasing order, ``[wire1, wire0]``,
    the chain runs the other way: the mode of ``wire1`` moves to ``wire0`` and the SPTM is the
    transpose. This makes the operation a cheap way to bring two distant modes next to each other on
    the NIF device.

    ``SptmFSwap`` is an alias of this class.

    Example:

    .. code-block:: python

        from matchcake.operations import SptmCompZX

        # Same SPTM as fSWAP(wires=[0, 1]) followed by fSWAP(wires=[1, 2]).
        sptm = SptmCompZX(wires=[0, 2]).sptm_matrix()
    """

    @classmethod
    def random(cls, wires: WiresLike, batch_size: Optional[int] = None, **kwargs: Any) -> "SptmCompZX":
        """Return the fermionic swap on the given wires.

        The gate has no parameters, so this is the same as calling the constructor. It exists so that
        the class can be used wherever a random SPTM operation is expected.

        :param wires: The two wires of the swap.
        :param batch_size: Ignored, since the gate has no parameters.
        :param kwargs: Additional keyword arguments passed to the constructor.
        :return: The fermionic swap operation.
        :rtype: SptmCompZX
        """
        return cls(wires=wires, **kwargs)

    def __init__(self, wires: Optional[WiresLike] = None, **kwargs: Any) -> None:
        """Build the SPTM of the fermionic swap.

        :param wires: The two wires of the swap. They can be non-adjacent, and their order sets the
            direction of the chain of swaps (see the class docstring).
        :param kwargs: Additional keyword arguments passed to
            :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`.
        """
        wires_arr = Wires(wires).toarray()
        wire0, wire1 = np.sort(wires_arr)
        size = 2 * (wire1 - wire0 + 1)
        matrix = np.zeros((size, size), dtype=int)

        first_idx = 2 * (wire1 - wire0)
        matrix[:2, first_idx : first_idx + 2] = np.eye(2)

        rows, cols = np.where(np.eye(size, k=-2))
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

    def to_matchgate(self) -> "CompZX":
        """Return the matchgate :class:`~matchcake.operations.CompZX` on the same wires.

        :return: The fSWAP matchgate.
        :rtype: CompZX
        """
        from ..fermionic_swap import CompZX

        return CompZX(wires=self.wires)


SptmFSwap = SptmCompZX
SptmFSwap.__name__ = "SptmFSwap"
