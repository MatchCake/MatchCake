import numbers
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operator
from pennylane.ops.functions import bind_new_parameters
from pennylane.ops.functions.equal import _equal_dispatch, _equal_operators
from pennylane.wires import Wires, WiresLike

from ...typing import TensorLike
from ...utils.covariance_matrix_toolbox import CovarianceMatrixToolbox
from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import infer_real_dtype
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

    A state without a definite fermion parity also has a displacement vector
    :math:`d_\mu = \mathrm{Tr}(\rho\, c_\mu)`, given with ``displacement_vector``. Such a state obeys
    Wick's theorem in the algebra extended by the parity mode, with the bordered matrix

    .. math::
        \widetilde\Lambda = \begin{pmatrix} \Lambda & d \\ -d^\top & 0 \end{pmatrix}.

    It arises as the reduced state of a matchgate circuit started from a product state with superposed
    qubits, on a block of wires that starts at the first wire of the register. The displacement vector is
    then the second parameter of the operation and gradients flow through it as well.

    A state obtained by tracing out wires that are not the last wires of a register also carries gate
    barriers, given with ``barriers`` and described in
    :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.barriers`. They are set by
    ``reduced_state`` and ``partial_trace`` of ``nif.qubit`` and by :func:`matchcake.reduced_state`, and
    they make the device refuse the gates and the observables whose result would not be the one of the
    reduced state of the qubits.

    The operation is supported natively by ``nif.qubit``. Its :meth:`decomposition` is a
    :class:`pennylane.QubitDensityMatrix` built from the dense density matrix, which can be queued on a
    mixed-state simulator such as ``default.mixed`` to get a dense reference for small systems. A state
    with a nonzero displacement vector or with barriers has no such decomposition.
    """

    num_wires = None
    grad_method = None
    ATOL = 1e-5

    @staticmethod
    def compute_decomposition(*params: TensorLike, wires: WiresLike, **hyperparameters: Any) -> List[Operator]:
        r"""
        Decompose into a :class:`pennylane.QubitDensityMatrix` holding the dense density matrix.

        :param params: The covariance matrix of shape ``(2n, 2n)``, optionally followed by the displacement
            vector of shape ``(2n,)``.
        :type params: TensorLike
        :param wires: Wires of the state.
        :type wires: WiresLike
        :param hyperparameters: The hyperparameters of the operation, which hold its barriers.
        :type hyperparameters: Any
        :return: The list with the density-matrix preparation.
        :rtype: List[Operator]
        :raises ValueError: If the covariance matrix is batched, since the density-matrix preparation has
            no batch dimension.
        :raises NotImplementedError: If the displacement vector is not zero or if the state has barriers.
        """
        covariance_matrix = params[0]
        if hyperparameters.get("barriers"):
            raise NotImplementedError(
                "A GaussianStatePrep with gate barriers cannot be decomposed into a qml.QubitDensityMatrix: "
                "a device that does not know the barriers would not refuse the gates and the observables "
                "that cross them. Use a device that supports GaussianStatePrep natively (nif.qubit)."
            )
        if len(params) > 1 and not GaussianStatePrep._is_zero(params[1]):
            raise NotImplementedError(
                "A GaussianStatePrep with a nonzero displacement vector cannot be decomposed into a "
                "qml.QubitDensityMatrix: its dense density matrix is not implemented. Use a device that "
                "supports GaussianStatePrep natively (nif.qubit)."
            )
        if len(qml.math.shape(covariance_matrix)) != 2:
            raise ValueError(
                "A batched GaussianStatePrep cannot be decomposed into a qml.QubitDensityMatrix, which has "
                "no batch dimension. Use a device that supports GaussianStatePrep natively."
            )
        return [qml.QubitDensityMatrix(CovarianceMatrixToolbox.to_density_matrix(covariance_matrix), wires=wires)]

    @staticmethod
    def _is_zero(displacement_vector: TensorLike) -> bool:
        r"""
        Tell whether a displacement vector vanishes within :attr:`DISPLACEMENT_ATOL`, the tolerance a
        device uses to tell whether a state has a displacement.

        :param displacement_vector: Displacement vector of shape ``(..., 2n)``.
        :type displacement_vector: TensorLike
        :return: ``True`` when every entry is zero within the tolerance.
        :rtype: bool
        """
        return bool(
            np.allclose(GaussianStatePrep._to_numpy(displacement_vector), 0.0, atol=GaussianStatePrep.DISPLACEMENT_ATOL)
        )

    @staticmethod
    def _prepare_barriers(barriers: Optional[Iterable[int]], n_wires: int) -> Tuple[int, ...]:
        r"""
        Check the positions of the barriers of a state and sort them.

        :param barriers: Positions of the barriers, or None for a state without barrier.
        :type barriers: Optional[Iterable[int]]
        :param odd_part_dropped: Whether the state stands for a state whose displacement vector was
            dropped, see
            :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.odd_part_dropped`. It is
            set by the reduced states of a device.
        :type odd_part_dropped: bool
        :param n_wires: Number of wires of the state.
        :type n_wires: int
        :return: The sorted positions, without repetition.
        :rtype: Tuple[int, ...]
        :raises ValueError: If a position is not an integer in ``[0, n_wires)``.
        """
        if barriers is None:
            return ()
        if isinstance(barriers, (str, bytes, numbers.Number)) or not isinstance(barriers, Iterable):
            raise ValueError(f"The barriers must be given as a sequence of wire positions, got {barriers!r}.")
        positions = list(barriers)
        for position in positions:
            is_integer = isinstance(position, numbers.Integral) and not isinstance(position, bool)
            if not is_integer or not 0 <= position < n_wires:
                raise ValueError(
                    f"A barrier must be the position of a wire of the state, an integer in [0, {n_wires}), "
                    f"got {position!r}."
                )
        return tuple(sorted({int(position) for position in positions}))

    @staticmethod
    def _to_numpy(tensor: TensorLike) -> np.ndarray:
        r"""
        Return the values of a tensor as a NumPy array, detached from any gradient graph.

        :param tensor: The tensor to convert.
        :type tensor: TensorLike
        :return: The values of the tensor.
        :rtype: np.ndarray
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(qml.math.toarray(tensor))

    @classmethod
    def _unflatten(cls, data, metadata) -> "GaussianStatePrep":
        r"""
        Recreate the operation from its parameters and its wires.

        :param data: The covariance matrix, optionally followed by the displacement vector.
        :param metadata: The wires and the hyperparameters of the operation.
        :return: The state preparation.
        :rtype: GaussianStatePrep
        """
        displacement_vector = data[1] if len(data) > 1 else None
        hyperparameters = dict(metadata[1]) if len(metadata) > 1 else {}
        return cls(
            data[0],
            wires=metadata[0],
            validate=False,
            displacement_vector=displacement_vector,
            barriers=hyperparameters.get("barriers"),
            odd_part_dropped=hyperparameters.get("odd_part_dropped", False),
        )

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
        *,
        displacement_vector: Optional[TensorLike] = None,
        barriers: Optional[Iterable[int]] = None,
        odd_part_dropped: bool = False,
    ):
        r"""
        :param covariance_matrix: Real antisymmetric matrix of shape ``(2n, 2n)`` or ``(B, 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param wires: Wires of the state.
        :type wires: WiresLike
        :param validate: Whether to check that the matrix is the covariance matrix of a physical state:
            real, antisymmetric and with normal-mode polarizations in :math:`[-1, 1]`. With a displacement
            vector, the polarizations are the ones of the bordered matrix.
        :type validate: bool
        :param displacement_vector: Real displacement vector of shape ``(2n,)`` or ``(B, 2n)``. Defaults to
            None, for a state with a definite fermion parity. When only one of the covariance matrix and
            the displacement vector is batched, the other one is repeated along the batch.
        :type displacement_vector: Optional[TensorLike]
        :param barriers: Positions of the gate barriers of the state in the order of ``wires``, see
            :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.barriers`. Defaults to
            None, for a state without barrier.
        :type barriers: Optional[Iterable[int]]
        :raises ValueError: If a shape does not match the wires, if a barrier is not the position of a
            wire, or if the validation fails.
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
        barriers = self._prepare_barriers(barriers, n_wires)
        self._has_displacement = displacement_vector is not None
        if displacement_vector is None:
            if validate:
                self._validate(covariance_matrix)
            super().__init__(covariance_matrix, wires=wires)
        else:
            covariance_matrix, displacement_vector = self._prepare_displacement(
                covariance_matrix, displacement_vector, validate
            )
            if validate:
                self._validate(covariance_matrix, displacement_vector)
            super().__init__(covariance_matrix, displacement_vector, wires=wires)
        self.hyperparameters["barriers"] = barriers
        self.hyperparameters["odd_part_dropped"] = bool(odd_part_dropped)

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
        fermionic modes of this state are the qubits of the same wires in the larger register. For the
        same reason the displacement vector keeps its entries on the wires of this state, and it vanishes
        on the added wires since :math:`\langle 0 | X | 0 \rangle = \langle 0 | Y | 0 \rangle = 0`.
        Gradients flow through the covariance matrix and the displacement vector.

        A barrier stays just before the wire it precedes. A wire added in the place of traced-out wires
        is therefore on the left of their barrier, and a wire added after the last wire is on the right
        of every barrier.

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
        extended_displacement = None
        if self._has_displacement:
            extended_displacement = qml.math.einsum("ai,...i->...a", embedding, self.displacement_vector)
        return GaussianStatePrep(
            extended,
            wires=wires,
            validate=False,
            displacement_vector=extended_displacement,
            barriers=[positions[barrier] for barrier in self.barriers],
            odd_part_dropped=self.odd_part_dropped,
        )

    def _prepare_displacement(
        self, covariance_matrix: TensorLike, displacement_vector: TensorLike, validate: bool
    ) -> Tuple[TensorLike, TensorLike]:
        r"""
        Bring a covariance matrix and a displacement vector to the same backend, real dtype and batch.

        Both become torch tensors when one of them is a torch tensor, so gradients keep flowing through
        the one that carries them.

        :param covariance_matrix: Covariance matrix of shape ``(2n, 2n)`` or ``(B, 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param displacement_vector: Displacement vector of shape ``(2n,)`` or ``(B, 2n)``.
        :type displacement_vector: TensorLike
        :param validate: Whether to check that the displacement vector is real.
        :type validate: bool
        :return: The covariance matrix and the displacement vector.
        :rtype: Tuple[TensorLike, TensorLike]
        :raises ValueError: If the shape of the displacement vector does not match the covariance matrix,
            or if it is not real while ``validate`` is ``True``.
        """
        displacement_vector = qml.math.asarray(displacement_vector)
        covariance_shape = tuple(qml.math.shape(covariance_matrix))
        shape = tuple(qml.math.shape(displacement_vector))
        n_majoranas = covariance_shape[-1]
        if len(shape) not in (1, 2) or shape[-1] != n_majoranas:
            raise ValueError(
                f"GaussianStatePrep expects a displacement vector of shape ({n_majoranas},) or its batched "
                f"form (B, {n_majoranas}), got shape {shape}."
            )
        if len(shape) == 2 and len(covariance_shape) == 3 and shape[0] != covariance_shape[0]:
            raise ValueError(
                f"The covariance matrix and the displacement vector must have the same batch size, got "
                f"{covariance_shape[0]} and {shape[0]}."
            )
        if "complex" in qml.math.get_dtype_name(displacement_vector):
            if validate and not np.allclose(self._to_numpy(displacement_vector).imag, 0.0, atol=self.ATOL):
                raise ValueError("The displacement vector must be real.")
            displacement_vector = qml.math.real(displacement_vector)
        if isinstance(covariance_matrix, torch.Tensor) or isinstance(displacement_vector, torch.Tensor):
            covariance_matrix = torch.as_tensor(covariance_matrix)
            displacement_vector = torch.as_tensor(displacement_vector).to(
                dtype=infer_real_dtype(covariance_matrix), device=covariance_matrix.device
            )
        else:
            displacement_vector = convert_and_cast_like(displacement_vector, qml.math.real(covariance_matrix))
        if len(shape) == 1 and len(covariance_shape) == 3:
            displacement_vector = qml.math.stack([displacement_vector] * covariance_shape[0])
        if len(shape) == 2 and len(covariance_shape) == 2:
            covariance_matrix = qml.math.stack([covariance_matrix] * shape[0])
        return covariance_matrix, displacement_vector

    def _validate(self, covariance_matrix: TensorLike, displacement_vector: Optional[TensorLike] = None) -> None:
        r"""
        Check that a matrix is the covariance matrix of a physical fermionic Gaussian state.

        With a displacement vector, the check is done on the bordered matrix
        :math:`\widetilde\Lambda`. A rotation of the extended algebra brings it to its normal modes, where
        the state is a product of single-mode mixtures, so its polarizations must lie in :math:`[-1, 1]`
        as well.

        :param covariance_matrix: Matrix of shape ``(..., 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param displacement_vector: Optional displacement vector of shape ``(..., 2n)``.
        :type displacement_vector: Optional[TensorLike]
        :raises ValueError: If the matrix is not real, not antisymmetric, or has a normal-mode
            polarization outside :math:`[-1, 1]`, within :attr:`ATOL`.
        """
        values = self._to_numpy(covariance_matrix)
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
        if displacement_vector is None:
            return
        displacement = self._to_numpy(displacement_vector)
        if not np.all(np.isfinite(displacement)):
            raise ValueError("The displacement vector must be finite.")
        n_majoranas = values.shape[-1]
        bordered = np.zeros(values.shape[:-2] + (n_majoranas + 1, n_majoranas + 1))
        bordered[..., :n_majoranas, :n_majoranas] = values
        bordered[..., :n_majoranas, n_majoranas] = displacement
        bordered[..., n_majoranas, :n_majoranas] = -displacement
        extended_polarizations = np.linalg.svd(bordered, compute_uv=False)
        if np.any(extended_polarizations > 1.0 + self.ATOL):
            raise ValueError(
                f"The covariance matrix and the displacement vector are not the ones of a physical state: "
                f"the normal-mode polarizations of the bordered matrix must lie in [-1, 1], got a maximal "
                f"magnitude of {float(np.max(extended_polarizations))}."
            )

    @property
    def barriers(self) -> Tuple[int, ...]:
        r"""
        Positions of the gate barriers of the prepared state, in the order of its wires.

        :return: The sorted positions of the barriers, empty for a state without barrier.
        :rtype: Tuple[int, ...]
        """
        return self.hyperparameters["barriers"]

    @property
    def odd_part_dropped(self) -> bool:
        r"""
        Whether the prepared state stands for a state whose displacement vector was dropped, see
        :attr:`~matchcake.operations.state_preparation.GaussianStatePrepBase.odd_part_dropped`.

        :return: ``True`` when a displacement vector was dropped.
        :rtype: bool
        """
        return self.hyperparameters["odd_part_dropped"]

    @property
    def covariance_matrix(self) -> TensorLike:
        r"""
        Majorana covariance matrix of the prepared state.

        :return: Real antisymmetric tensor of shape ``(2n, 2n)``, or ``(B, 2n, 2n)`` for a batched state.
        :rtype: TensorLike
        """
        return self.data[0]

    @property
    def displacement_vector(self) -> TensorLike:
        r"""
        Displacement vector :math:`d_\mu = \mathrm{Tr}(\rho\, c_\mu)` of the prepared state.

        :return: Real tensor of shape ``(2n,)``, or ``(B, 2n)`` for a batched state. It is zero when the
            state was built without a displacement vector.
        :rtype: TensorLike
        """
        if self._has_displacement:
            return self.data[1]
        return super().displacement_vector

    @property
    def num_params(self) -> int:
        return 2 if self._has_displacement else 1

    @property
    def ndim_params(self) -> tuple:
        return (2, 1) if self._has_displacement else (2,)


@bind_new_parameters.register(GaussianStatePrep)
def _bind_new_parameters_gaussian_state_prep(op, params):
    """Reconstruct a GaussianStatePrep with new parameters, the displacement vector being the second one."""
    displacement_vector = params[1] if len(params) > 1 else None
    return GaussianStatePrep(
        params[0],
        wires=op.wires,
        validate=False,
        displacement_vector=displacement_vector,
        barriers=op.barriers,
        odd_part_dropped=op.odd_part_dropped,
    )


@_equal_dispatch.register(GaussianStatePrep)
def _equal_gaussian_state_prep(op1, op2, **kwargs):
    """
    Compare two GaussianStatePrep for :func:`pennylane.equal`.

    The default comparison pairs the parameters of the two operations and ignores the ones without a
    partner, so it would not see the displacement vector that only one of them carries. The barriers
    are compared here as well.
    """
    if op1.num_params != op2.num_params:
        return "op1 and op2 have different numbers of parameters: only one of them has a displacement vector."
    if op1.barriers != op2.barriers:
        return f"op1 and op2 have different barriers. Got {op1.barriers} and {op2.barriers}."
    return _equal_operators(op1, op2, **kwargs)
