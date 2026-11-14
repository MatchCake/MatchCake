from typing import Any, Optional

import numpy as np
from pennylane.wires import WiresLike

from ...utils import make_wires_continuous
from .single_particle_transition_matrix import SingleParticleTransitionMatrixOperation


class SptmFSwapHH(SingleParticleTransitionMatrixOperation):
    r"""
    Single-particle transition matrix (SPTM) of a layer of fermionic swaps followed by
    :class:`~matchcake.operations.CompHH` gates.

    The wires are made continuous (from the smallest to the largest wire) and grouped in pairs
    :math:`(w_0, w_0 + 1), (w_0 + 2, w_0 + 3), \dots`. On each pair, the operation applies
    :class:`~matchcake.operations.fSWAP` followed by :class:`~matchcake.operations.CompHH`, whose SPTM is

    .. math::
        R = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0 \\
            0 & -1 & 0 & 0
        \end{bmatrix}.

    With an odd number of wires, the last wire has no partner and is left unchanged, so the SPTM is
    orthogonal for any number of wires.
    """

    @classmethod
    def random(cls, wires: WiresLike, batch_size: Optional[int] = None, **kwargs: Any) -> "SptmFSwapHH":
        """Return the operation on the given wires.

        The operation has no parameters, so this is the same as calling the constructor.

        :param wires: The wires of the operation.
        :param batch_size: Ignored, since the operation has no parameters.
        :param kwargs: Additional keyword arguments passed to the constructor.
        :return: The operation.
        :rtype: SptmFSwapHH
        """
        return cls(wires=wires, **kwargs)

    def __init__(self, wires: Optional[WiresLike] = None, **kwargs: Any) -> None:
        """Build the SPTM of the layer.

        :param wires: The wires of the operation. They are made continuous, from the smallest to the
            largest wire.
        :param kwargs: Additional keyword arguments passed to
            :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`.
        """
        all_wires = make_wires_continuous(wires)
        n_wires = len(all_wires)
        matrix = np.zeros((2 * n_wires, 2 * n_wires), dtype=int)
        matrix[..., ::2, ::2] = np.eye(n_wires)
        matrix[..., np.arange(1, 2 * n_wires - 2, 4), np.arange(3, 2 * n_wires, 4)] = 1
        matrix[..., np.arange(3, 2 * n_wires, 4), np.arange(1, 2 * n_wires - 2, 4)] = -1
        if n_wires % 2 == 1:
            # The last wire has no partner: identity on its odd Majorana mode.
            matrix[..., 2 * n_wires - 1, 2 * n_wires - 1] = 1
        super().__init__(matrix, wires=wires, **kwargs)
