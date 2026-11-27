from typing import Any, Callable, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
import torch
from pennylane.exceptions import MatrixUndefinedError, PowUndefinedError
from pennylane.math import expm
from pennylane.operation import Operation, Operator, StatePrepBase
from pennylane.ops.op_math import Adjoint, Pow
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
from scipy.linalg import schur as scipy_schur

from ... import utils
from ...utils import make_wires_continuous
from ...utils.logm import logm
from ...utils.majorana import MajoranaGetter
from ...utils.math import (
    convert_and_cast_like,
    dagger,
    fermionic_operator_matmul,
    orthonormalize,
)


class SingleParticleTransitionMatrixOperation(Operation):
    r"""
    Represents a single-particle transition matrix (SPTM) operation.

    This class defines and implements the behavior of single-particle transition
    matrix operations in a quantum computation framework. A SPTM is defined as

    .. math::
        R = \exp(4 h)

    where :math:`R` is the SPTM and :math:`h` is the free fermionic integrals, or transition energies between
    orbitals. The free fermions hamiltonian is then defines as

    .. math::
        H = i \sum_{i,j=0}^{2N - 1} h_{i,j} c_i c_j

    where :math:`c_k` is the k-th majorana operator and :math:`N` is the number of orbitals/qubits/wires.

    In terms of matchgate, the SPTM is defined as

    .. math::
        R_{\mu\nu} = \frac{1}{4} \text{Tr}((U c_\mu U^\dagger)c_\nu)

    where :math:`R_{\mu\nu}` are the matrix elements of the SPTM and :math:`U` is a matchgate.
    """

    num_wires = None
    num_params = 1
    par_domain = "A"
    grad_method = "A"
    grad_recipe = None
    generator = None

    casting_priorities: List[Literal["numpy", "autograd", "jax", "tf", "torch"]] = [
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",  # greater index means higher priority
    ]
    DEFAULT_CHECK_MATRIX = False
    has_matrix = False  # matrix() raises, the stored matrix is the SPTM and not the qubit unitary

    ALLOWED_ANGLES: Optional[List[float]] = None
    EQUAL_ALLOWED_ANGLES: Optional[List[float]] = None
    DEFAULT_CHECK_ANGLES = False
    DEFAULT_CLIP_ANGLES = True
    DEFAULT_NORMALIZE = False
    DEFAULT_CHECK_REAL = True
    REAL_PART_ATOL = 1e-6
    MINUS_ONE_EIGVAL_ATOL = 1e-8
    # Set by the module of DisjointSptmOperation, which cannot be imported from here.
    DISJOINT_PRODUCT_CLASS: Any = None

    @staticmethod
    def make_wires_continuous(wires: Wires):
        wires_array = wires.tolist()
        min_wire, max_wire = min(wires_array), max(wires_array)
        return Wires(range(min_wire, max_wire + 1))

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Optional[WiresLike] = None,
        **hyperparameters: dict[str, Any],
    ):
        unitary = SingleParticleTransitionMatrixOperation.to_unitary_matrix(params[0])
        decomposition_wires = Wires(wires)
        if 2 * len(decomposition_wires) != qml.math.shape(params[0])[-1]:
            # An SPTM on integer wires with gaps is stored on their continuous span.
            decomposition_wires = make_wires_continuous(decomposition_wires)
        return [qml.QubitUnitary(unitary, wires=decomposition_wires)]

    @staticmethod
    def _is_integer_exponent(z: Any) -> bool:
        """Return True when ``z`` is an integer or an integer-valued float (booleans excluded).

        :param z: The exponent.
        :type z: Any
        :return: True if ``z`` is an integer power.
        :rtype: bool
        """
        if isinstance(z, (bool, np.bool_)):
            return False
        if isinstance(z, (int, np.integer)):
            return True
        return isinstance(z, (float, np.floating)) and float(z).is_integer()

    @staticmethod
    def _real_logm_from_schur(matrix: np.ndarray) -> np.ndarray:
        r"""Return a real antisymmetric logarithm of a real orthogonal matrix with determinant one.

        The real Schur form of an orthogonal matrix is block diagonal, with 2 by 2 rotation blocks and
        1 by 1 blocks equal to :math:`\pm 1`. Rotation blocks map to their angle, the :math:`+1`
        blocks to zero, and the :math:`-1` blocks are paired into planes rotated by :math:`\pi`. This
        stays real where the principal logarithm assigns :math:`i\pi` to every :math:`-1` eigenvalue.

        :param matrix: Real orthogonal matrix, shape (2n, 2n).
        :type matrix: np.ndarray
        :return: Real antisymmetric matrix ``log`` such that ``expm(log) == matrix``, shape (2n, 2n).
        :rtype: np.ndarray
        :raises ValueError: If ``matrix`` has an odd number of :math:`-1` eigenvalues (determinant -1).
        """
        atol = SingleParticleTransitionMatrixOperation.MINUS_ONE_EIGVAL_ATOL
        schur_form, schur_vectors = scipy_schur(matrix, output="real")
        size = matrix.shape[-1]
        log_form = np.zeros((size, size))
        minus_one_indices = []
        index = 0
        while index < size:
            if index + 1 < size and abs(schur_form[index + 1, index]) > atol:
                angle = np.arctan2(schur_form[index + 1, index], schur_form[index, index])
                log_form[index + 1, index] = angle
                log_form[index, index + 1] = -angle
                index += 2
                continue
            if schur_form[index, index] < 0:
                minus_one_indices.append(index)
            index += 1
        if len(minus_one_indices) % 2 != 0:
            raise ValueError(
                "The single-particle transition matrix has an odd number of -1 eigenvalues, so its "
                "determinant is -1 and it has no real logarithm."
            )
        for first_index, second_index in zip(minus_one_indices[::2], minus_one_indices[1::2]):
            log_form[second_index, first_index] = np.pi
            log_form[first_index, second_index] = -np.pi
        return schur_vectors @ log_form @ schur_vectors.T

    @staticmethod
    def _orthogonal_logm(matrix: TensorLike) -> TensorLike:
        r"""Return a real antisymmetric logarithm of a (batch of) SPTM(s).

        The differentiable principal logarithm :func:`logm` is used, except for matrices with a real
        negative eigenvalue (an exact :math:`-1`, or a :math:`-1` pair split by rounding), where it is
        complex and not a valid generator. Those are computed with :meth:`_real_logm_from_schur`,
        which is not differentiable, and are replaced by the identity before calling :func:`logm`.

        :param matrix: The SPTM(s), shape (..., 2n, 2n).
        :type matrix: TensorLike
        :return: The logarithm(s), shape (..., 2n, 2n), with the backend of :func:`logm`.
        :rtype: TensorLike
        """
        shape = qml.math.shape(matrix)
        numpy_batch = np.real(np.asarray(qml.math.to_numpy(matrix))).reshape(-1, *shape[-2:])
        atol = SingleParticleTransitionMatrixOperation.MINUS_ONE_EIGVAL_ATOL
        needs_schur = np.array(
            [
                np.any((np.real(eigvals) < 0) & (np.abs(np.imag(eigvals)) < atol))
                for eigvals in (np.linalg.eigvals(sub) for sub in numpy_batch)
            ]
        )
        if not np.any(needs_schur):
            return logm(matrix)
        mask = qml.math.real(convert_and_cast_like(needs_schur.reshape(*shape[:-2], 1, 1), matrix)) > 0
        identity = convert_and_cast_like(np.eye(shape[-1]), matrix)
        log_matrix = logm(qml.math.where(mask, identity, matrix))
        replacement = np.zeros((len(numpy_batch), *shape[-2:]))
        for batch_index in np.flatnonzero(needs_schur):
            replacement[batch_index] = SingleParticleTransitionMatrixOperation._real_logm_from_schur(
                numpy_batch[batch_index]
            )
        replacement = convert_and_cast_like(replacement.reshape(shape), log_matrix)
        mask = qml.math.real(convert_and_cast_like(needs_schur.reshape(*shape[:-2], 1, 1), log_matrix)) > 0
        return qml.math.where(mask, replacement, log_matrix)

    @staticmethod
    def _enforce_real(matrix: TensorLike, check_real: bool) -> TensorLike:
        """Return the real part of ``matrix``, keeping it real-valued (a floating-point dtype).

        Real inputs are returned unchanged. Complex inputs (e.g. a matrix built from a matchgate)
        have their imaginary part dropped via the differentiable ``qml.math.real``, so gradients flow
        through. When ``check_real`` is True, the imaginary part must be negligible (within
        :attr:`REAL_PART_ATOL`) or a ``ValueError`` is raised.

        :param matrix: The candidate single-particle transition matrix.
        :type matrix: TensorLike
        :param check_real: Whether to verify that a complex input has a negligible imaginary part.
        :type check_real: bool
        :return: The real-valued matrix.
        :rtype: TensorLike
        :raises ValueError: If ``check_real`` is True and the imaginary part exceeds ``REAL_PART_ATOL``.
        """
        if "complex" not in qml.math.get_dtype_name(matrix).lower():
            return matrix
        if check_real:
            atol = SingleParticleTransitionMatrixOperation.REAL_PART_ATOL
            max_abs_imaginary = qml.math.max(qml.math.abs(qml.math.imag(matrix)))
            if max_abs_imaginary > atol:
                raise ValueError(
                    f"The single-particle transition matrix must be real, but its maximum absolute "
                    f"imaginary part is {max_abs_imaginary}, which exceeds {atol}."
                )
        return qml.math.real(matrix)

    @staticmethod
    def _writable_like(matrix: TensorLike, blocks: Sequence[TensorLike]) -> Optional[TensorLike]:
        """Return ``matrix`` in the backend, dtype and batch shape its product with ``blocks`` has.

        The matrix is returned as it is when it already has them, and as a new array otherwise. Torch
        has priority over numpy, as in :attr:`casting_priorities`. The dtype is the promotion of all the
        dtypes, and a batched block makes the matrix batched.

        :param matrix: The matrix whose columns are about to be updated, shape (..., 2n, 2n).
        :type matrix: TensorLike
        :param blocks: The blocks it is multiplied by, each of shape (..., 2k, 2k).
        :type blocks: Sequence[TensorLike]
        :return: The matrix to write into, or None when a backend other than numpy and torch is
            involved, which cannot be written in place.
        :rtype: Optional[TensorLike]
        """
        interfaces = {qml.math.get_interface(tensor) for tensor in (matrix, *blocks)}
        if not interfaces <= {"numpy", "torch"}:
            return None
        batch_sizes = [qml.math.shape(block)[0] for block in blocks if qml.math.ndim(block) > 2]
        if "torch" in interfaces:
            reference = next(tensor for tensor in (matrix, *blocks) if isinstance(tensor, torch.Tensor))
            dtype = torch.as_tensor(matrix).dtype
            for block in blocks:
                dtype = torch.promote_types(dtype, torch.as_tensor(block).dtype)
            matrix = torch.as_tensor(matrix, device=reference.device).to(dtype)
            if batch_sizes and matrix.ndim == 2:
                matrix = matrix.expand(batch_sizes[0], *matrix.shape).clone()
            return matrix
        numpy_dtype = np.result_type(matrix, *blocks)
        if matrix.dtype != numpy_dtype:
            matrix = matrix.astype(numpy_dtype)
        if batch_sizes and matrix.ndim == 2:
            matrix = np.broadcast_to(matrix, (batch_sizes[0], *matrix.shape)).copy()
        return matrix

    @staticmethod
    def _block_like(block: TensorLike, matrix: TensorLike) -> TensorLike:
        """Return ``block`` in the backend, dtype and device of ``matrix``, keeping its gradient.

        :param block: A block, shape (..., 2k, 2k).
        :type block: TensorLike
        :param matrix: A numpy array or a torch tensor.
        :type matrix: TensorLike
        :return: The block, shape (..., 2k, 2k).
        :rtype: TensorLike
        """
        if isinstance(matrix, torch.Tensor):
            return torch.as_tensor(block, device=matrix.device).to(matrix.dtype)
        return np.asarray(block, dtype=matrix.dtype)

    @staticmethod
    def multiply_columns(matrix: TensorLike, blocks: Sequence[Tuple[int, TensorLike]]) -> Optional[TensorLike]:
        r"""Multiply a matrix on the right by blocks acting on disjoint ranges of columns.

        The product of ``matrix`` with a block padded with the identity only changes the columns the
        block acts on,

        .. math::
            R_{:, [s, s + 2k)} \leftarrow R_{:, [s, s + 2k)} B,

        which costs :math:`O(n k^2)` instead of the :math:`O(n^3)` of the padded product. Blocks of the
        same size are applied in one batched product. The columns are written in place, so ``matrix``
        must not be shared with a caller that still needs its previous value, see
        :meth:`_writable_like` for when a new array is returned instead.

        :param matrix: The matrix, shape (2n, 2n) or (b, 2n, 2n), a numpy array or a torch tensor.
        :type matrix: TensorLike
        :param blocks: ``(start, block)`` pairs: the first column each block acts on and the block,
            shape (2k, 2k) or (b, 2k, 2k). The column ranges must not overlap.
        :type blocks: Sequence[Tuple[int, TensorLike]]
        :return: The product, shape (2n, 2n) or (b, 2n, 2n), or None when a backend other than numpy
            and torch is involved.
        :rtype: Optional[TensorLike]
        """
        matrix = SingleParticleTransitionMatrixOperation._writable_like(matrix, [block for _, block in blocks])
        if matrix is None:
            return None
        if isinstance(matrix, torch.Tensor) and matrix.is_leaf and matrix.requires_grad:
            matrix = matrix.clone()
        by_size: dict = {}
        for start, block in blocks:
            block = SingleParticleTransitionMatrixOperation._block_like(block, matrix)
            by_size.setdefault(block.shape[-1], []).append((start, block))
        for size, group in by_size.items():
            if len(group) == 1:
                start, block = group[0]
                columns = matrix[..., :, start : start + size]
                if isinstance(columns, torch.Tensor):
                    # autograd keeps the columns for the gradient of the block, they are overwritten below
                    columns = columns.clone()
                matrix[..., :, start : start + size] = columns @ block
                continue
            # indices: (n_blocks, 2k), columns: (..., 2n, n_blocks, 2k)
            indices: Any = np.asarray([start for start, _ in group])[:, None] + np.arange(size)[None, :]
            if isinstance(matrix, torch.Tensor):
                indices = torch.as_tensor(indices, device=matrix.device)
            if all(block.ndim == 2 for _, block in group):
                stacked = qml.math.stack([block for _, block in group])  # (n_blocks, 2k, 2k)
            else:
                batch_shape = (matrix.shape[0], size, size)
                stacked = qml.math.stack(  # (b, n_blocks, 2k, 2k)
                    [
                        block.expand(batch_shape)
                        if isinstance(block, torch.Tensor)
                        else np.broadcast_to(block, batch_shape)
                        for _, block in group
                    ],
                    axis=1,
                )
            columns = matrix[..., :, indices]
            if stacked.ndim == 3:
                matrix[..., :, indices] = qml.math.einsum("...nls,lst->...nlt", columns, stacked)
            else:
                matrix[..., :, indices] = qml.math.einsum("bnls,blst->bnlt", columns, stacked)
        return matrix

    @classmethod
    def clip_angles(cls, angles):
        """
        If the ALLOWED_ANGLES is not none, set the angles to the closest allowed angle.
        If all the angles are equal in the last dimension, the EQUAL_ALLOWED_ANGLES is used instead.
        """
        if cls.ALLOWED_ANGLES is None and cls.EQUAL_ALLOWED_ANGLES is None:
            return angles

        real_angles = qml.math.real(angles)
        angles = qml.math.where(real_angles >= 0, real_angles % (2 * np.pi), real_angles % (-2 * np.pi))
        angles_shape = qml.math.shape(angles)
        if len(angles_shape) > 0:
            angles_flatten = qml.math.reshape(angles, (-1, angles_shape[-1]))
        else:
            angles_flatten = qml.math.reshape(angles, (-1, 1))
        equal_mask = qml.math.all(qml.math.isclose(angles_flatten[..., 0, None], angles_flatten[..., :]), -1)
        allowed_clipped_angles = cls.clip_to_allowed_angles(angles_flatten, cls.ALLOWED_ANGLES)
        equal_allowed_clipped_angles = cls.clip_to_allowed_angles(angles_flatten, cls.EQUAL_ALLOWED_ANGLES)
        angles_flatten = qml.math.where(equal_mask[..., None], equal_allowed_clipped_angles, allowed_clipped_angles)
        angles = qml.math.reshape(angles_flatten, angles_shape)
        return angles

    @classmethod
    def clip_to_allowed_angles(cls, angles, allowed_angles: Optional[Sequence[float]] = None):
        """
        If the ALLOWED_ANGLES is not none, set the angles to the closest allowed angle.
        """
        if allowed_angles is None:
            return angles  # pragma: no cover

        angles = qml.math.where(angles >= 0, angles % (2 * np.pi), angles % (-2 * np.pi))
        allowed_angles_array = convert_and_cast_like(np.array(allowed_angles), angles)
        angles_shape = qml.math.shape(angles)
        angles_flatten = qml.math.reshape(angles, (-1, 1))
        distances = allowed_angles_array - angles_flatten
        abs_distances = qml.math.abs(distances)
        min_distances = distances[np.arange(distances.shape[0]), qml.math.argmin(abs_distances, -1)]
        angles_flatten = angles_flatten.squeeze() + min_distances
        angles = qml.math.reshape(angles_flatten, angles_shape)
        return angles

    @classmethod
    def check_angles(cls, angles):
        """
        If the ALLOWED_ANGLES is not none, check if the angles are in the allowed range.
        """
        if cls.ALLOWED_ANGLES is None and cls.EQUAL_ALLOWED_ANGLES is None:
            return True

        angles = qml.math.where(angles >= 0, angles % (2 * np.pi), angles % (-2 * np.pi))
        angles_shape = qml.math.shape(angles)
        if len(angles_shape) > 0:
            angles_flatten = qml.math.reshape(angles, (-1, angles_shape[-1]))
        else:
            angles_flatten = qml.math.reshape(angles, (-1, 1))
        equal_mask = qml.math.all(qml.math.isclose(angles_flatten[..., 0, None], angles_flatten[..., :]), -1)

        not_equal_angles = angles_flatten[~equal_mask].reshape(-1)
        if not qml.math.all(qml.math.isin(not_equal_angles, cls.ALLOWED_ANGLES)):
            raise ValueError(f"Invalid angles: {angles}. Expected: {cls.ALLOWED_ANGLES}")

        equal_angles = angles_flatten[equal_mask].reshape(-1)
        if not qml.math.all(qml.math.isin(equal_angles, cls.EQUAL_ALLOWED_ANGLES)):
            raise ValueError(f"Invalid angles: {angles}. Expected: {cls.EQUAL_ALLOWED_ANGLES}")

        return True

    @classmethod
    def from_circuit(
        cls, circuit: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> "SingleParticleTransitionMatrixOperation":
        """
        Contract every operation of a quantum function into a single SPTM operation.

        The quantum function is recorded with ``*args`` and ``**kwargs`` (its measurements are ignored),
        and its operations are multiplied in circuit order, the same way the
        :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` applies them. The result
        acts on the continuous span of the wires used by the circuit, so it can be merged with other
        circuits on disjoint wires using :meth:`from_operations` and chained with ``@``. When called
        inside a QNode, only the returned operation is queued, so the circuit is applied once as a
        single gate.

        :param circuit: The quantum function to contract.
        :type circuit: Callable[..., Any]
        :param args: Positional arguments passed to ``circuit``.
        :type args: Any
        :param kwargs: Keyword arguments passed to ``circuit``.
        :type kwargs: Any
        :return: The SPTM of the whole circuit, shape (..., 2n, 2n).
        :rtype: SingleParticleTransitionMatrixOperation
        :raises ValueError: If the circuit contains a state preparation or no operation.
        """
        # Only the returned operation is queued, like any other operation constructor.
        with qml.QueuingManager.stop_recording():
            tape = qml.tape.make_qscript(circuit)(*args, **kwargs)
            ops = [op for op in tape.operations if not isinstance(op, qml.Identity)]
            state_preps = [op for op in ops if isinstance(op, StatePrepBase)]
            if state_preps:
                raise ValueError(
                    f"Cannot contract the state preparation(s) {state_preps} into a {cls.__name__}, "
                    f"since it only represents a free fermionic evolution."
                )
            if len(ops) == 0:
                raise ValueError(f"Cannot build a {cls.__name__} from a circuit without operations.")
            sptm_ops = [cls.from_operation(op) for op in ops]
            all_wires = cls.make_wires_continuous(Wires.all_wires([op.wires for op in sptm_ops], sort=True))
            matrices = [op.pad(all_wires).sptm_matrix() for op in sptm_ops]
            matrices = utils.math.convert_tensors_to_same_type_and_cast_to(matrices, cls.casting_priorities)
            matrix = matrices[0]
            for next_matrix in matrices[1:]:
                matrix = fermionic_operator_matmul(matrix, next_matrix, operator="einsum")
        return SingleParticleTransitionMatrixOperation(matrix, wires=all_wires)

    @classmethod
    def from_operation(
        cls, op: Union[Operator, "SingleParticleTransitionMatrixOperation"], **kwargs
    ) -> "SingleParticleTransitionMatrixOperation":
        """
        Creates an instance of SingleParticleTransitionMatrixOperation from the given operation.

        This class method allows the construction of a
        SingleParticleTransitionMatrixOperation instance from various types of
        operators. If the input operator is already an instance of
        SingleParticleTransitionMatrixOperation, it is directly returned.
        It also checks if the provided operator has the `to_sptm_operation` method or
        the `single_particle_transition_matrix` attribute. If it's the case, the SPTM will be
        constructed from `to_sptm_operation` or from `single_particle_transition_matrix` if the first
        doesn't exist. A lazy :class:`~pennylane.ops.op_math.Adjoint` wrapper is converted by
        converting its base and taking the adjoint of the resulting SPTM, and a
        :class:`~pennylane.ops.op_math.Pow` wrapper with an integer exponent by raising the converted
        base SPTM to that power. When none of the required attributes or
        methods are present, we try to build a MatchGateOperation from the given operation matrix.
        If nothing works, a ValueError is raised.

        :param op: The input operator to be converted. It can be an instance of
                   `Operator` or `SingleParticleTransitionMatrixOperation`.
        :param kwargs: Any additional keyword arguments passed to the method.
        :return: A new instance of SingleParticleTransitionMatrixOperation.
        :rtype: SingleParticleTransitionMatrixOperation
        :raises ValueError: When the input operator cannot be converted due to
                            missing attributes or methods.
        """
        from ..matchgate_operation import MatchgateOperation

        if isinstance(op, SingleParticleTransitionMatrixOperation):
            return op
        if isinstance(op, Adjoint):
            return cls.from_operation(op.base, **kwargs).adjoint()
        if isinstance(op, Pow) and cls._is_integer_exponent(op.z):
            return cls.from_operation(op.base, **kwargs)._integer_power(op.z)
        if hasattr(op, "to_sptm_operation") and callable(op.to_sptm_operation):
            return op.to_sptm_operation()
        if hasattr(op, "single_particle_transition_matrix"):
            return SingleParticleTransitionMatrixOperation(
                op.single_particle_transition_matrix, wires=op.wires, **kwargs
            )
        try:
            return MatchgateOperation(op.matrix(), wires=op.wires, **kwargs).to_sptm_operation()
        except Exception as e:
            raise ValueError(
                f"Cannot convert {type(op)} to {cls.__name__} "
                f"without the attribute 'single_particle_transition_matrix' or the method 'to_sptm_operation'."
            ) from e

    @classmethod
    def from_operations(
        cls,
        ops: Iterable[Union[Any, "SingleParticleTransitionMatrixOperation"]],
        **kwargs,
    ) -> "Optional[SingleParticleTransitionMatrixOperation]":
        """
        This method will contract multiple SingleParticleTransitionMatrixOperations into a single one.
        Each operation must act on a different set of wires. The operations are kept as separate
        blocks, see :class:`DisjointSptmOperation`, and are only written into one dense matrix when
        that matrix is asked for.

        :param ops: The operations to contract.
        :param kwargs: Additional keyword arguments.

        :return: The contracted SingleParticleTransitionMatrixOperation.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        ops = list(ops)
        if len(ops) == 0:
            return None
        ops = [cls.from_operation(op, **kwargs) for op in ops]
        if len(ops) == 1:
            return ops[0]

        return cls.DISJOINT_PRODUCT_CLASS(ops, **kwargs)

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        wires = kwargs.get("wires", None)
        assert wires is not None, "wires kwarg must be set."
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        params_indexes = np.triu_indices(2 * len(wires), k=1)
        rn_params = rn_gen.normal(size=(([batch_size] if batch_size is not None else []) + [params_indexes[0].size]))
        matrix = np.zeros(
            ([batch_size] if batch_size is not None else []) + [2 * len(wires), 2 * len(wires)], dtype=complex
        )
        matrix[..., params_indexes[0], params_indexes[1]] = rn_params
        matrix[..., params_indexes[1], params_indexes[0]] = -rn_params
        exp_matrix = expm(4 * matrix)
        return exp_matrix

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(
            cls.random_params(batch_size=batch_size, wires=wires, **kwargs),
            wires=wires,
            **kwargs,
        )

    def __init__(
        self,
        matrix: TensorLike,
        wires: Optional[Union[Sequence[int], Wires]] = None,
        *,
        clip_angles: bool = DEFAULT_CLIP_ANGLES,
        check_angles: bool = DEFAULT_CHECK_ANGLES,
        check_matrix: bool = DEFAULT_CHECK_MATRIX,
        check_real: bool = DEFAULT_CHECK_REAL,
        normalize: bool = DEFAULT_NORMALIZE,
        **kwargs,
    ):
        """
        Initialize an operation that applies a single-particle transition represented by the
        specified matrix with optional parameters for clipping angles, checking matrix and
        angle validity, and normalization.

        The single-particle transition matrix is a real orthogonal matrix, so the stored matrix is
        always real (a floating-point dtype). A complex matrix may be passed (for example, one built
        from a matchgate), in which case its imaginary part is dropped after an optional check that it
        is negligible. Dropping the imaginary part is differentiable, so gradients still flow through a
        complex input.

        :param matrix: A tensor-like object defining the transition matrix.
        :param wires: Optional; Specifies the wires or subsystems the operation acts on. Can
            be a sequence of integers or a `Wires` object.
        :param clip_angles: Boolean flag indicating whether to clip angles in the matrix.
            Defaults to `DEFAULT_CLIP_ANGLES`.
        :param check_angles: Boolean flag indicating whether to validate the angles in the
            matrix. Defaults to `DEFAULT_CHECK_ANGLES`.
        :param check_matrix: Boolean flag indicating whether to check if the matrix lies
            within the SO(4) group. Defaults to `DEFAULT_CHECK_MATRIX`.
        :param check_real: Boolean flag indicating whether to verify that a complex input matrix has a
            negligible imaginary part (within `REAL_PART_ATOL`) before dropping it.
            Defaults to `DEFAULT_CHECK_REAL`.
        :param normalize: Boolean flag indicating whether to orthonormalize the matrix
            prior to initialization. Defaults to `DEFAULT_NORMALIZE`.
        :param kwargs: Additional keyword arguments passed to parent classes or methods.
        :raises ValueError: If `check_matrix` is True and the provided matrix does not belong
            to the SO(4) group, or if `check_real` is True and the imaginary part of `matrix`
            exceeds `REAL_PART_ATOL`.
        """
        if normalize:
            matrix = orthonormalize(matrix)
        matrix = self._enforce_real(matrix, check_real)
        self._matrix = matrix
        self._wires = wires
        super().__init__(matrix, wires=wires)
        self._hyperparameters = {
            "clip_angles": clip_angles,
            "check_angles": check_angles,
            "check_matrix": check_matrix,
            "check_real": check_real,
        }
        if check_matrix:
            if not self.check_is_in_so4():
                raise ValueError(f"Matrix is not in SO(4): {matrix}")
        self.kwargs = kwargs

    def matrix(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        r"""
        Refuse to return a qubit matrix for this operation.

        PennyLane's :meth:`~pennylane.operation.Operator.matrix` contract is the
        :math:`2^n \times 2^n` qubit unitary, while this operation only stores its
        :math:`2n \times 2n` single-particle transition matrix. Raising here makes generic
        PennyLane code (``qml.matrix``, ``default.qubit``, lazy wrappers such as ``qml.adjoint``
        or ``qml.pow``) fall back to :meth:`compute_decomposition` instead of silently treating the
        SPTM as a unitary. Use :meth:`sptm_matrix` to get the SPTM and :meth:`to_qubit_unitary` to
        get the qubit unitary.

        :param wire_order: Ignored.
        :type wire_order: Optional[WiresLike]
        :raises MatrixUndefinedError: Always.
        """
        raise MatrixUndefinedError(
            f"{self.__class__.__name__} does not define a qubit matrix. Use sptm_matrix() for the "
            f"single-particle transition matrix or to_qubit_unitary() for the qubit unitary."
        )

    def sptm_matrix(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        """
        Return the single-particle transition matrix, padded with the identity to ``wire_order``.

        :param wire_order: The wires to pad the SPTM to. Defaults to the operation's wires.
        :type wire_order: Optional[WiresLike]
        :return: The SPTM, shape (..., 2n, 2n) where n is the number of continuous wires.
        :rtype: TensorLike
        """
        wires = Wires(self.wires) if wire_order is None else Wires(wire_order)
        return self.pad(wires)._matrix

    def check_is_in_so4(self, atol=1e-6, rtol=1e-6):
        matrix = self.sptm_matrix()
        if self.batch_size is None:
            matrix = matrix[None, ...]
        for sub_matrix in matrix:
            if not np.isclose(np.linalg.det(sub_matrix), 1, atol=atol, rtol=rtol):
                return False
            if not np.allclose(np.linalg.inv(sub_matrix), sub_matrix.T, atol=atol, rtol=rtol):
                return False
        return True

    def check_is_unitary(self, atol=1e-6, rtol=1e-6):
        matrix = self.sptm_matrix()
        if self.batch_size is None:
            matrix = matrix[None, ...]
        eye = np.eye(matrix.shape[-1])
        expected_eye = qml.math.einsum("...ij,...jk->...ik", dagger(matrix), matrix)
        return np.allclose(expected_eye, eye, atol=atol, rtol=rtol)

    def pad(self, wires: Wires):
        if not isinstance(wires, Wires):
            wires = Wires(wires)
        cs_wires = make_wires_continuous(wires)
        if self.cs_wires == cs_wires:
            return self
        matrix = self._matrix
        if qml.math.ndim(matrix) == 2:
            padded_matrix = np.eye(2 * len(cs_wires))
        elif qml.math.ndim(matrix) == 3:
            padded_matrix = np.zeros((qml.math.shape(matrix)[0], 2 * len(cs_wires), 2 * len(cs_wires)))
            padded_matrix[:, ...] = np.eye(2 * len(cs_wires))
        else:
            raise NotImplementedError("This method is not implemented yet.")
        padded_matrix = utils.math.convert_and_cast_like(padded_matrix, matrix)
        wire0_idx = cs_wires.index(self.cs_wires[0])
        slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-2])
        slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-1])
        try:
            padded_matrix[..., slice_0, slice_1] = matrix
        except Exception:
            padded_matrix[..., slice_0, slice_1] = utils.math.convert_and_cast_like(matrix, padded_matrix)
        kwargs = self._hyperparameters.copy()
        return SingleParticleTransitionMatrixOperation(padded_matrix, wires=cs_wires, **kwargs)

    def __matmul__(self, other):
        if not isinstance(other, SingleParticleTransitionMatrixOperation):
            raise ValueError(f"Cannot multiply {self.__class__.__name__} with {type(other)}")

        all_wires = Wires.all_wires([self.wires, other.wires], sort=True)
        wires = self.make_wires_continuous(all_wires)

        padded = self.pad(wires)._matrix
        if qml.math.get_interface(padded) in ("numpy", "torch"):
            # pad returns the stored matrix itself when the wires are already the right ones
            padded = padded.clone() if isinstance(padded, torch.Tensor) else np.array(padded)
        matrix = self.multiply_columns(padded, other.column_blocks(wires))
        if matrix is None:
            matrix = qml.math.einsum("...ij,...jk->...ik", padded, other.pad(wires).sptm_matrix())
        return SingleParticleTransitionMatrixOperation(matrix, wires=wires, **self._hyperparameters)

    def __imatmul__(self, other):
        """Multiply this operation on the right by ``other``, writing into its own matrix when possible.

        When ``other`` acts within the wires of this operation, only the columns it acts on are
        updated, in place, and this operation is returned. Any tensor that was read from this
        operation before then holds the new values too. Otherwise the result is that of ``self @ other``.

        :param other: The operation to multiply by.
        :type other: SingleParticleTransitionMatrixOperation
        :return: The product.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        if not isinstance(other, SingleParticleTransitionMatrixOperation):
            raise ValueError(f"Cannot multiply {self.__class__.__name__} with {type(other)}")
        wires = self.make_wires_continuous(Wires.all_wires([self.wires, other.wires], sort=True))
        if wires != self.cs_wires or self.blocks is not None:
            return self @ other
        matrix = self.multiply_columns(self._matrix, other.column_blocks(wires))
        if matrix is None:
            return self @ other
        self._matrix = matrix
        self.data = (matrix,)
        return self

    def column_blocks(self, wires: Wires) -> List[Tuple[int, TensorLike]]:
        """Return the blocks of this operation with the first column each acts on in a larger matrix.

        :param wires: The continuous wires of the larger matrix, which must contain those of this operation.
        :type wires: Wires
        :return: ``(start, block)`` pairs, the blocks having shape (..., 2k, 2k).
        :rtype: List[Tuple[int, TensorLike]]
        """
        return [(2 * wires.index(self.cs_wires[0]), self._matrix)]

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return SingleParticleTransitionMatrixOperation(
            dagger(self.sptm_matrix()), wires=self.wires, **self._hyperparameters
        )

    def pow(self, z: float) -> List[Operator]:
        """
        Decompose this operation raised to the power ``z``.

        Integer powers (including integer-valued floats and numpy integers) give a single SPTM
        operation whose matrix is the SPTM raised to ``z``. A negative power uses the transpose,
        which is the inverse of an SPTM. Other powers are not defined, since a fractional power of an
        SPTM depends on the branch of its logarithm.

        :param z: The exponent.
        :type z: float
        :return: A single operation equal to this operation raised to the power ``z``.
        :rtype: List[Operator]
        :raises PowUndefinedError: If ``z`` is not an integer.
        """
        with qml.QueuingManager.stop_recording():
            powered = self._integer_power(z)
        if qml.QueuingManager.recording():
            powered = qml.apply(powered)
        return [powered]

    def to_cuda(self):  # pragma: no cover
        """Return the same operation with its SPTM on the GPU.

        Integer matrices (as for the fermionic swap or the identity) are cast to float64, since the
        batched matrix products of the NIF device are not implemented for integer tensors on CUDA.

        :return: The operation on the GPU.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        from ...utils import torch_utils  # pragma: no cover

        matrix = torch_utils.to_cuda(self.sptm_matrix(), dtype=None)  # pragma: no cover
        if not (matrix.is_floating_point() or matrix.is_complex()):  # pragma: no cover
            matrix = matrix.to(torch.float64)  # pragma: no cover
        return SingleParticleTransitionMatrixOperation(  # pragma: no cover
            matrix,  # pragma: no cover
            wires=self.wires,  # pragma: no cover
            **self._hyperparameters,  # pragma: no cover
        )  # pragma: no cover

    def to_torch(self):
        from ...utils import torch_utils

        return SingleParticleTransitionMatrixOperation(
            torch_utils.to_tensor(self.sptm_matrix(), dtype=None),
            wires=self.wires,
            **self._hyperparameters,
        )

    def __round__(self, n=None):
        return SingleParticleTransitionMatrixOperation(
            qml.math.round(self.sptm_matrix(), n), wires=self.wires, **self._hyperparameters
        )

    def real(self):
        return SingleParticleTransitionMatrixOperation(
            qml.math.real(self.sptm_matrix()), wires=self.wires, **self._hyperparameters
        )

    def __trunc__(self):
        return SingleParticleTransitionMatrixOperation(
            qml.math.trunc(self.sptm_matrix()), wires=self.wires, **self._hyperparameters
        )

    def to_matchgate(self):
        from ..matchgate_operation import MatchgateOperation

        unitary = self.to_unitary_matrix(self.sptm_matrix())
        return MatchgateOperation(unitary, wires=self.wires, **self.hyperparameters)

    def to_qubit_unitary(self) -> qml.QubitUnitary:
        """
        Converts the internal matrix representation to a `qml.QubitUnitary` object. The method takes the unitary matrix
        representation of the object and associates it with the specified wires. This is primarily used for creating
        quantum operations that act on qubits in Pennylane.

        :return: A `qml.QubitUnitary` object that represents the unitary operation acting on the specified wires.
        :rtype: qml.QubitUnitary
        """
        return qml.QubitUnitary(self.to_unitary_matrix(self.sptm_matrix()), wires=self.wires, unitary_check=True)

    def to_qubit_operation(self) -> Operation:
        """
        Converts the current object into a qubit operation.

        This method determines the appropriate qubit operation based on the number
        of wires in the object. If the number of wires is two, it converts the object to
        a matchgate operation. Otherwise, it falls back to converting the object into
        a generic qubit unitary operation.

        :return: An instance of `Operation` representing either a match gate or a
            qubit unitary operation, depending on the number of wires.
        :rtype: Operation
        """
        if len(self.wires) == 2:
            return self.to_matchgate()
        return self.to_qubit_unitary()

    @staticmethod
    def to_unitary_matrix(matrix: TensorLike) -> TensorLike:
        r"""
        Compute the Matchgate unitary matrix from the single particle transition matrix.

        ... :math:
            U = \exp(-\sum\limits_{i,j=0}^{2N-1} h_{ij} c_i c_j)
            R = \exp(4 h) \implies h = \frac{1}{4} \log(R)

        where :math:`c_i` are the Majorana operators, :math:`R` is the current single particle transition matrix.
        """
        wires = np.arange(matrix.shape[-1] // 2)
        majorana_tensor = MajoranaGetter(n=len(wires)).majorana_tensor
        h = (1 / 4) * SingleParticleTransitionMatrixOperation._orthogonal_logm(matrix)
        unitary = qml.math.expm(
            -1.0
            * qml.math.einsum(
                "...ij,ikq,jqp->...kp",
                h,
                majorana_tensor,
                majorana_tensor,
                optimize="optimal",
            )
        )
        return unitary

    def _integer_power(self, z: float) -> "SingleParticleTransitionMatrixOperation":
        """
        Return this operation raised to an integer power as a single SPTM operation.

        :param z: The exponent, an integer or an integer-valued float.
        :type z: float
        :return: The SPTM operation whose matrix is the SPTM raised to ``z``, shape (..., 2n, 2n).
        :rtype: SingleParticleTransitionMatrixOperation
        :raises PowUndefinedError: If ``z`` is not an integer.
        """
        if not self._is_integer_exponent(z):
            raise PowUndefinedError(f"{self.__class__.__name__} only supports integer powers, got {z}.")
        exponent = int(round(float(z)))
        matrix = self.sptm_matrix() if exponent >= 0 else dagger(self.sptm_matrix())
        powered = qml.math.linalg.matrix_power(matrix, abs(exponent))
        return SingleParticleTransitionMatrixOperation(powered, wires=self.wires, **self._hyperparameters)

    @property
    def blocks(self) -> Optional[List["SingleParticleTransitionMatrixOperation"]]:
        """The operations on disjoint wires this operation is the product of, if it is kept as such.

        :return: None for an operation stored as one matrix.
        :rtype: Optional[List[SingleParticleTransitionMatrixOperation]]
        """
        return None

    @property
    def single_particle_transition_matrix(self) -> TensorLike:
        """
        The single-particle transition matrix on the operation's own wires.

        Same name as :attr:`MatchgateOperation.single_particle_transition_matrix`, so generic code
        can read the SPTM of either operation family.

        :return: The SPTM, shape (..., 2n, 2n).
        :rtype: TensorLike
        """
        return self._matrix

    @property
    def shape(self):
        return qml.math.shape(self._matrix)

    @property
    def dtype(self) -> str:
        """The backend-agnostic dtype name of the underlying matrix.

        :return: The dtype name (e.g. ``"complex128"``).
        :rtype: str
        """
        return qml.math.get_dtype_name(self._matrix)

    @property
    def batch_size(self):
        if qml.math.ndim(self._matrix) > 2:
            return self.shape[0]
        return None

    @property
    def sorted_wires(self):
        return Wires(sorted(self.wires.tolist()))

    @property
    def cs_wires(self):
        return Wires(make_wires_continuous(self.wires))
