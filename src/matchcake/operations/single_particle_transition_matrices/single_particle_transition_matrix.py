from typing import Any, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np
import pennylane as qml
from pennylane.math import expm
from pennylane.operation import AnyWires, Operation, Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from ... import utils
from ...utils import make_wires_continuous
from ...utils.logm import logm
from ...utils.majorana import MajoranaGetter
from ...utils.math import (
    convert_and_cast_like,
    dagger,
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

    num_wires = AnyWires
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

    ALLOWED_ANGLES = None
    EQUAL_ALLOWED_ANGLES = None
    DEFAULT_CHECK_ANGLES = False
    DEFAULT_CLIP_ANGLES = True
    DEFAULT_NORMALIZE = False

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
        return [qml.QubitUnitary(unitary, wires=wires)]

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
        doesn't exist. When none of the required attributes or
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
    ) -> "SingleParticleTransitionMatrixOperation":
        """
        This method will contract multiple SingleParticleTransitionMatrixOperations into a single one.
        Each operation must act on a different set of wires.

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

        all_wires = Wires.all_wires([op.cs_wires for op in ops], sort=True)
        all_wires = cls.make_wires_continuous(all_wires)
        batch_sizes = [op.batch_size for op in ops if op.batch_size is not None] + [None]
        batch_size = batch_sizes[0]
        if batch_size is None:
            matrix = np.eye(2 * len(all_wires), dtype=complex)
        else:
            matrix = np.zeros((batch_size, 2 * len(all_wires), 2 * len(all_wires)), dtype=complex)
            matrix[:, ...] = np.eye(2 * len(all_wires), dtype=matrix.dtype)

        ops_sptms = [op.matrix() for op in ops]
        ops_sptms = utils.math.convert_tensors_to_same_type_and_cast_to(
            ops_sptms, cls.casting_priorities, dtype=complex
        )
        matrix = utils.math.convert_like_and_cast_to(matrix, ops_sptms[0], dtype=complex)

        for op, op_matrix in zip(ops, ops_sptms):
            wire0_idx = all_wires.index(op.sorted_wires[0])
            slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + op_matrix.shape[-2])
            slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + op_matrix.shape[-1])
            matrix[..., slice_0, slice_1] = utils.math.convert_and_cast_like(op_matrix, matrix)
        return SingleParticleTransitionMatrixOperation(matrix, wires=all_wires, **kwargs)

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
        id=None,
        clip_angles: bool = DEFAULT_CLIP_ANGLES,
        check_angles: bool = DEFAULT_CHECK_ANGLES,
        check_matrix: bool = DEFAULT_CHECK_MATRIX,
        normalize: bool = DEFAULT_NORMALIZE,
        **kwargs,
    ):
        """
        Initialize an operation that applies a single-particle transition represented by the
        specified matrix with optional parameters for clipping angles, checking matrix and
        angle validity, and normalization.

        :param matrix: A tensor-like object defining the transition matrix.
        :param wires: Optional; Specifies the wires or subsystems the operation acts on. Can
            be a sequence of integers or a `Wires` object.
        :param id: Optional; The operation's unique identifier.
        :param clip_angles: Boolean flag indicating whether to clip angles in the matrix.
            Defaults to `DEFAULT_CLIP_ANGLES`.
        :param check_angles: Boolean flag indicating whether to validate the angles in the
            matrix. Defaults to `DEFAULT_CHECK_ANGLES`.
        :param check_matrix: Boolean flag indicating whether to check if the matrix lies
            within the SO(4) group. Defaults to `DEFAULT_CHECK_MATRIX`.
        :param normalize: Boolean flag indicating whether to orthonormalize the matrix
            prior to initialization. Defaults to `DEFAULT_NORMALIZE`.
        :param kwargs: Additional keyword arguments passed to parent classes or methods.
        :raises ValueError: If `check_matrix` is True and the provided matrix does not belong
            to the SO(4) group.
        """
        if normalize:
            matrix = orthonormalize(matrix)
        self._matrix = matrix
        self._wires = wires
        Operation.__init__(self, matrix, wires=wires, id=id)
        self._hyperparameters = {
            "clip_angles": clip_angles,
            "check_angles": check_angles,
            "check_matrix": check_matrix,
        }
        if check_matrix:
            if not self.check_is_in_so4():
                raise ValueError(f"Matrix is not in SO(4): {matrix}")

    def matrix(self, wire_order=None) -> TensorLike:
        wires = Wires(self.wires) if wire_order is None else Wires(wire_order)
        return self.pad(wires)._matrix

    def check_is_in_so4(self, atol=1e-6, rtol=1e-6):
        matrix = self.matrix()
        if self.batch_size is None:
            matrix = matrix[None, ...]
        for sub_matrix in matrix:
            if not np.isclose(np.linalg.det(sub_matrix), 1, atol=atol, rtol=rtol):
                return False
            if not np.allclose(np.linalg.inv(sub_matrix), sub_matrix.T, atol=atol, rtol=rtol):
                return False
        return True

    def check_is_unitary(self, atol=1e-6, rtol=1e-6):
        matrix = self.matrix()
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
        except:
            padded_matrix[..., slice_0, slice_1] = utils.math.convert_and_cast_like(matrix, padded_matrix)
        kwargs = self._hyperparameters.copy()
        return SingleParticleTransitionMatrixOperation(padded_matrix, wires=cs_wires, **kwargs)

    def __matmul__(self, other):
        if not isinstance(other, SingleParticleTransitionMatrixOperation):
            raise ValueError(f"Cannot multiply {self.__class__.__name__} with {type(other)}")

        all_wires = Wires.all_wires([self.wires, other.wires], sort=True)
        wires = self.make_wires_continuous(all_wires)

        _self = self.pad(wires).matrix()
        other = self.from_operation(other, **self._hyperparameters).pad(wires).matrix()

        return SingleParticleTransitionMatrixOperation(
            qml.math.einsum("...ij,...jk->...ik", _self, other),
            wires=wires,
            **self._hyperparameters,
        )

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return SingleParticleTransitionMatrixOperation(dagger(self.matrix()), wires=self.wires, **self._hyperparameters)

    def to_cuda(self):
        import torch

        from ...utils import torch_utils

        return SingleParticleTransitionMatrixOperation(
            torch_utils.to_cuda(self.matrix(), dtype=torch.complex128),
            wires=self.wires,
            **self._hyperparameters,
        )

    def to_torch(self):
        import torch

        from ...utils import torch_utils

        return SingleParticleTransitionMatrixOperation(
            torch_utils.to_tensor(self.matrix(), dtype=torch.complex128),
            wires=self.wires,
            **self._hyperparameters,
        )

    def __round__(self, n=None):
        return SingleParticleTransitionMatrixOperation(
            qml.math.round(self.matrix(), n), wires=self.wires, **self._hyperparameters
        )

    def real(self):
        return SingleParticleTransitionMatrixOperation(
            qml.math.real(self.matrix()), wires=self.wires, **self._hyperparameters
        )

    def __trunc__(self):
        return SingleParticleTransitionMatrixOperation(
            qml.math.trunc(self.matrix()), wires=self.wires, **self._hyperparameters
        )

    def to_matchgate(self):
        from ..matchgate_operation import MatchgateOperation

        unitary = self.to_unitary_matrix(self.matrix())
        return MatchgateOperation(unitary, wires=self.wires, id=self.id, **self.hyperparameters)

    def to_qubit_unitary(self) -> qml.QubitUnitary:
        """
        Converts the internal matrix representation to a `qml.QubitUnitary` object. The method takes the unitary matrix
        representation of the object and associates it with the specified wires. This is primarily used for creating
        quantum operations that act on qubits in Pennylane.

        :return: A `qml.QubitUnitary` object that represents the unitary operation acting on the specified wires.
        :rtype: qml.QubitUnitary
        """
        return qml.QubitUnitary(self.to_unitary_matrix(self.matrix()), wires=self.wires, id=self.id, unitary_check=True)

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
        majorana_getter = MajoranaGetter(n=len(wires))
        majorana_tensor = qml.math.stack([majorana_getter[i] for i in range(2 * majorana_getter.n)])
        h = (1 / 4) * logm(matrix)
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

    @property
    def shape(self):
        return qml.math.shape(self._matrix)

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
