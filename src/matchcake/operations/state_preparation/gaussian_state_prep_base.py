from abc import abstractmethod
from typing import Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import StatePrepBase
from pennylane.wires import WiresLike

from ...typing import TensorLike
from ...utils.torch_utils import infer_real_dtype, to_numpy


class GaussianStatePrepBase(StatePrepBase):
    r"""
    Base class of the state preparations that a non-interacting fermionic device evolves natively.

    Such a state is described by its Majorana covariance matrix

    .. math::
        (\Lambda_0)_{\mu\nu} = i\, \mathrm{Tr}(\rho\, c_\mu c_\nu), \quad \mu \neq \nu,

    and its displacement vector :math:`d_\mu = \mathrm{Tr}(\rho\, c_\mu)`, with the Jordan-Wigner Majorana
    operators :math:`c_{2k} = Z_{<k} X_k` and :math:`c_{2k+1} = Z_{<k} Y_k`. A matchgate circuit with
    single-particle transition matrix :math:`R` maps them to :math:`R^\top \Lambda_0 R` and
    :math:`R^\top d`, which is all the device needs to compute expectation values, probabilities and
    samples.

    Subclasses provide :attr:`covariance_matrix`, :meth:`to_wires` and, when the state is not an even
    fermionic Gaussian state, :attr:`displacement_vector`. A state obtained by tracing out wires can
    also have :attr:`barriers`.

    :cvar DISPLACEMENT_ATOL: Absolute tolerance under which a displacement vector is considered zero.
    """

    DISPLACEMENT_ATOL = 1e-12

    @abstractmethod
    def to_wires(self, wires: WiresLike) -> "GaussianStatePrepBase":
        r"""
        Return the same state on ``wires``, with the wires this state does not act on in :math:`|0\rangle`.

        A device uses this to extend a state preparation on part of its wires to its full register.

        :param wires: The wires of the new state. They must contain every wire of this state.
        :type wires: WiresLike
        :return: This state if ``wires`` already matches its wires, otherwise a new state on ``wires``.
        :rtype: GaussianStatePrepBase
        """

    def is_displaced(self) -> bool:
        r"""
        Tell whether the prepared state has a nonzero displacement vector.

        A matchgate circuit rotates the displacement vector by an orthogonal matrix, so a state stays
        displaced, or not displaced, through such a circuit. A batched state is displaced when one of its
        elements is.

        :return: ``True`` when an entry of :attr:`displacement_vector` exceeds :attr:`DISPLACEMENT_ATOL`
            in magnitude.
        :rtype: bool
        """
        return not np.allclose(to_numpy(self.displacement_vector), 0.0, atol=self.DISPLACEMENT_ATOL)

    @property
    def barriers(self) -> Tuple[int, ...]:
        r"""
        Positions of the gate barriers of the prepared state, in the order of its wires.

        A barrier at position :math:`p` sits just before the wire at position :math:`p`. It marks wires
        that were traced out there, and whose Jordan-Wigner strings are still hidden in the Majorana
        operators of the wires at positions :math:`\geq p`. The expectation value of a Pauli word is the
        one of the reduced state of the qubits exactly when, for every barrier, the word has an even
        number of :math:`X` and :math:`Y` on the wires at positions :math:`\geq p`. A device therefore
        refuses the gates that act on both sides of a barrier and the observables that break this rule.

        :return: The sorted positions of the barriers, empty by default.
        :rtype: Tuple[int, ...]
        """
        return ()

    @property
    def odd_part_dropped(self) -> bool:
        r"""
        Whether the prepared state stands for a state whose displacement vector was dropped.

        A reduced state loses its displacement vector when wires are traced out before its first wire:
        its Pauli words with an odd number of :math:`X` and :math:`Y` then depend on the removed wires.
        The barrier left there must be kept as long as a wire remains on its right, even after wires are
        added and removed again on its left.

        :return: ``True`` when a displacement vector was dropped, ``False`` by default.
        :rtype: bool
        """
        return False

    @property
    @abstractmethod
    def covariance_matrix(self) -> TensorLike:
        r"""
        Majorana covariance matrix of the prepared state.

        :return: Real antisymmetric tensor of shape ``(2n, 2n)``, or ``(B, 2n, 2n)`` for a batched state.
        :rtype: TensorLike
        """

    @property
    def displacement_vector(self) -> TensorLike:
        r"""
        Displacement vector :math:`d_\mu = \mathrm{Tr}(\rho\, c_\mu)` of the prepared state.

        It vanishes for a state with a definite fermion parity, which is the default.

        :return: Real tensor of shape ``(2n,)``, or ``(B, 2n)`` for a batched state.
        :rtype: TensorLike
        """
        covariance_matrix = self.covariance_matrix
        device = covariance_matrix.device if isinstance(covariance_matrix, torch.Tensor) else None
        return torch.zeros(
            tuple(qml.math.shape(covariance_matrix)[:-1]), dtype=infer_real_dtype(covariance_matrix), device=device
        )
