from typing import List, Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator
from pennylane.wires import Wires, WiresLike

from ...typing import TensorLike
from ...utils.covariance_matrix_toolbox import CovarianceMatrixToolbox
from ...utils.math import convert_and_cast_like
from .gaussian_state_prep_base import GaussianStatePrepBase


class GaussianStatePrep(GaussianStatePrepBase):
    r"""
    Prepare a fermionic Gaussian state, possibly mixed, from its Majorana covariance matrix.

    The state is the even Gaussian state :math:`\rho` whose only non-trivial correlations are

    .. math::
        \Lambda_{\mu\nu} = i\, \mathrm{Tr}(\rho\, c_\mu c_\nu), \quad \mu \neq \nu,

    with the Jordan-Wigner Majorana operators :math:`c_{2k} = Z_{<k} X_k` and
    :math:`c_{2k+1} = Z_{<k} Y_k` of its wires. It is pure when :math:`\Lambda^2 = -\mathbb{1}` and mixed
    otherwise. Mixed Gaussian states arise as the reduced state of a matchgate circuit on a contiguous
    block of its wires (see :func:`matchcake.covariance_matrix`), as thermal states of free-fermion
    Hamiltonians and as inputs with classical uncertainty on the occupation of the modes.

    The covariance matrix is given as a ``(2n, 2n)`` matrix, or as a ``(B, 2n, 2n)`` batch of matrices.
    Gradients flow through it.

    The operation is supported natively by ``nif.qubit``. Its :meth:`decomposition` is a
    :class:`pennylane.QubitDensityMatrix` built from the dense density matrix, which can be queued on a
    mixed-state simulator such as ``default.mixed`` to get a dense reference for small systems.
    """

    num_wires = None
    grad_method = None
    ATOL = 1e-5

    @staticmethod
    def compute_decomposition(covariance_matrix: TensorLike, wires: WiresLike) -> List[Operator]:
        r"""
        Decompose into a :class:`pennylane.QubitDensityMatrix` holding the dense density matrix.

        :param covariance_matrix: Covariance matrix of shape ``(2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param wires: Wires of the state.
        :type wires: WiresLike
        :return: The list with the density-matrix preparation.
        :rtype: List[Operator]
        :raises ValueError: If the covariance matrix is batched, since the density-matrix preparation has
            no batch dimension.
        """
        if len(qml.math.shape(covariance_matrix)) != 2:
            raise ValueError(
                "A batched GaussianStatePrep cannot be decomposed into a qml.QubitDensityMatrix, which has "
                "no batch dimension. Use a device that supports GaussianStatePrep natively."
            )
        return [qml.QubitDensityMatrix(CovarianceMatrixToolbox.to_density_matrix(covariance_matrix), wires=wires)]

    @classmethod
    def from_polarizations(
        cls,
        polarizations: TensorLike,
        wires: WiresLike,
        sptm: Optional[TensorLike] = None,
    ) -> "GaussianStatePrep":
        r"""
        Construct the state from the polarizations of its normal modes.

        The state is the product of the single-mode mixtures
        :math:`\tfrac{1 + \lambda_j}{2} |0\rangle\langle 0| + \tfrac{1 - \lambda_j}{2} |1\rangle\langle 1|`,
        optionally evolved by the Gaussian unitary of single-particle transition matrix :math:`R`. Its
        covariance matrix is :math:`R^\top \Lambda_D R` with
        :math:`(\Lambda_D)_{2j, 2j+1} = -\lambda_j`.

        :param polarizations: Polarizations :math:`\lambda_j = \langle Z_j \rangle \in [-1, 1]` of shape
            ``(n,)`` or ``(B, n)``.
        :type polarizations: TensorLike
        :param wires: Wires of the state.
        :type wires: WiresLike
        :param sptm: Single-particle transition matrix :math:`R` of shape ``(2n, 2n)`` or ``(B, 2n, 2n)``.
            Defaults to the identity.
        :type sptm: Optional[TensorLike]
        :return: The Gaussian state preparation.
        :rtype: GaussianStatePrep
        """
        covariance_matrix = CovarianceMatrixToolbox.from_polarizations(polarizations)
        if sptm is not None:
            sptm = convert_and_cast_like(sptm, covariance_matrix)
            covariance_matrix = qml.math.einsum("...ij,...ik,...kl->...jl", sptm, covariance_matrix, sptm)
        return cls(covariance_matrix, wires=wires)

    def __init__(
        self,
        covariance_matrix: TensorLike,
        wires: WiresLike,
        validate: bool = True,
    ):
        r"""
        :param covariance_matrix: Real antisymmetric matrix of shape ``(2n, 2n)`` or ``(B, 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param wires: Wires of the state.
        :type wires: WiresLike
        :param validate: Whether to check that the matrix is the covariance matrix of a physical state:
            real, antisymmetric and with normal-mode polarizations in :math:`[-1, 1]`.
        :type validate: bool
        :raises ValueError: If the shape does not match the wires or if the validation fails.
        """
        wires = Wires(wires)
        covariance_matrix = qml.math.asarray(covariance_matrix)
        n_wires = len(wires)
        shape = tuple(qml.math.shape(covariance_matrix))
        if len(shape) not in (2, 3) or shape[-2:] != (2 * n_wires, 2 * n_wires):
            raise ValueError(
                f"GaussianStatePrep expects a covariance matrix of shape (2 * n_wires, 2 * n_wires) = "
                f"({2 * n_wires}, {2 * n_wires}) or its batched form (B, {2 * n_wires}, {2 * n_wires}), "
                f"got shape {shape}."
            )
        if validate:
            self._validate(covariance_matrix)
        super().__init__(covariance_matrix, wires=wires)

    def state_vector(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        r"""
        A Gaussian state given by its covariance matrix is mixed in general and has no state vector.

        :param wire_order: Unused.
        :type wire_order: Optional[WiresLike]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{type(self).__name__} prepares a state that is mixed in general and has no state vector. Use a "
            f"device that supports it natively (nif.qubit), or queue its decomposition on a mixed-state "
            f"simulator (default.mixed)."
        )

    def to_wires(self, wires: WiresLike) -> "GaussianStatePrep":
        r"""
        Return the same state on ``wires``, with the wires this state does not act on in :math:`|0\rangle`.

        The wires of this state must appear in ``wires`` in the same relative order. The Jordan-Wigner
        strings that run through the added wires then act on :math:`|0\rangle` and are trivial, so the
        fermionic modes of this state are the qubits of the same wires in the larger register. Gradients
        flow through the covariance matrix.

        :param wires: The wires of the new state. They must contain every wire of this state.
        :type wires: WiresLike
        :return: This state if ``wires`` already matches its wires, otherwise a new
            :class:`GaussianStatePrep` on ``wires``.
        :rtype: GaussianStatePrep
        :raises ValueError: If ``wires`` misses a wire of this state or lists them in another order.
        """
        wires = Wires(wires)
        if wires == self.wires:
            return self
        missing_wires = [wire for wire in self.wires if wire not in wires]
        if missing_wires:
            raise ValueError(f"The wires {wires.tolist()} do not contain the wires {missing_wires} of {self}.")
        positions = [wires.index(wire) for wire in self.wires]
        if positions != sorted(positions):
            raise ValueError(
                f"The wires {self.wires.tolist()} of {type(self).__name__} must be in the same order as in "
                f"{wires.tolist()}. Reordering fermionic modes is not a relabeling of qubits: apply the "
                f"state on wires in register order and move the modes with fermionic swaps instead."
            )
        covariance_matrix = self.covariance_matrix
        n_total = len(wires)
        indices = CovarianceMatrixToolbox.majorana_indices(positions)
        embedding = np.zeros((2 * n_total, len(indices)))
        embedding[indices, np.arange(len(indices))] = 1.0
        vacuum_polarizations = np.ones(n_total)
        vacuum_polarizations[positions] = 0.0
        embedding = convert_and_cast_like(embedding, covariance_matrix)
        vacuum = convert_and_cast_like(
            CovarianceMatrixToolbox.from_polarizations(vacuum_polarizations), covariance_matrix
        )
        extended = qml.math.einsum("ai,...ij,bj->...ab", embedding, covariance_matrix, embedding) + vacuum
        return GaussianStatePrep(extended, wires=wires, validate=False)

    def _validate(self, covariance_matrix: TensorLike) -> None:
        r"""
        Check that a matrix is the covariance matrix of a physical fermionic Gaussian state.

        :param covariance_matrix: Matrix of shape ``(..., 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :raises ValueError: If the matrix is not real, not antisymmetric, or has a normal-mode
            polarization outside :math:`[-1, 1]`, within :attr:`ATOL`.
        """
        if isinstance(covariance_matrix, torch.Tensor):
            values = covariance_matrix.detach().cpu().numpy()
        else:
            values = np.asarray(qml.math.toarray(covariance_matrix))
        if np.iscomplexobj(values) and not np.allclose(values.imag, 0.0, atol=self.ATOL):
            raise ValueError("The covariance matrix must be real.")
        values = np.real(values)
        if not np.allclose(values, -np.swapaxes(values, -1, -2), atol=self.ATOL):
            raise ValueError("The covariance matrix must be antisymmetric.")
        polarizations = np.asarray(CovarianceMatrixToolbox.mode_polarizations(values))
        if np.any(polarizations > 1.0 + self.ATOL):
            raise ValueError(
                f"The covariance matrix is not the one of a physical state: its normal-mode polarizations "
                f"must lie in [-1, 1], got a maximal magnitude of {float(np.max(polarizations))}."
            )

    @property
    def covariance_matrix(self) -> TensorLike:
        r"""
        Majorana covariance matrix of the prepared state.

        :return: Real antisymmetric tensor of shape ``(2n, 2n)``, or ``(B, 2n, 2n)`` for a batched state.
        :rtype: TensorLike
        """
        return self.data[0]

    @property
    def num_params(self) -> int:
        return 1

    @property
    def ndim_params(self) -> tuple:
        return (2,)
