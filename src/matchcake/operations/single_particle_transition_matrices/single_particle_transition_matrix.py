from typing import Any, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.operation import AnyWires, Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import utils
from ...constants import _CIRCUIT_MATMUL_DIRECTION
from ...utils import make_wires_continuous
from ...utils.math import (
    circuit_matmul,
    convert_and_cast_like,
    dagger,
    det,
    fermionic_operator_matmul,
    orthonormalize,
)
from ...utils.torch_utils import detach


class _SingleParticleTransitionMatrix:
    casting_priorities = [
        "numpy",
        "autograd",
        "jax",
        "tf",
        "torch",
    ]  # greater index means higher priority

    @staticmethod
    def make_wires_continuous(wires: Wires):
        wires_array = wires.tolist()
        min_wire, max_wire = min(wires_array), max(wires_array)
        return Wires(range(min_wire, max_wire + 1))

    @classmethod
    def from_operation(
        cls, op: Union[Any, "_SingleParticleTransitionMatrix"], **kwargs
    ) -> "_SingleParticleTransitionMatrix":
        if isinstance(op, cls):
            return op
        return cls(op.single_particle_transition_matrix, wires=op.wires, **kwargs)

    @classmethod
    def from_operations(
        cls, ops: Iterable[Union[Any, "_SingleParticleTransitionMatrix"]]
    ) -> "_SingleParticleTransitionMatrix":
        from ..matchgate_operation import MatchgateOperation

        ops = list(ops)
        if len(ops) == 0:
            return None
        if len(ops) == 1:
            return cls.from_operation(ops[0])
        all_wires = Wires.all_wires([op.wires for op in ops], sort=True)
        all_wires = cls.make_wires_continuous(all_wires)
        batch_sizes = [op.batch_size for op in ops if op.batch_size is not None] + [None]
        batch_size = batch_sizes[0]
        if batch_size is None:
            matrix = pnp.eye(2 * len(all_wires), dtype=complex)
        else:
            matrix = pnp.zeros((batch_size, 2 * len(all_wires), 2 * len(all_wires)), dtype=complex)
            matrix[:, ...] = pnp.eye(2 * len(all_wires), dtype=matrix.dtype)

        for op in ops:
            wire0_idx = all_wires.index(op.wires[0])

            if isinstance(op, MatchgateOperation):
                sptm = cls.from_operation(op)
            elif isinstance(op, cls):
                sptm = op
            else:
                raise ValueError(f"Cannot convert {type(op)} to {cls.__name__}.")

            slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + sptm.shape[-2])
            slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + sptm.shape[-1])
            # matrix[..., slice_0, slice_1] = qml.math.einsum(
            #     "...ij,...jk->...ik",
            #     detach(matrix[..., slice_0, slice_1]),
            #     utils.math.convert_and_cast_like(sptm, matrix)
            # )
            matrix[..., slice_0, slice_1] = utils.math.convert_and_cast_like(sptm, matrix)
        return cls(matrix, wires=all_wires)

    @classmethod
    def from_spt_matrices(
        cls, matrices: Iterable["_SingleParticleTransitionMatrix"]
    ) -> "_SingleParticleTransitionMatrix":
        matrices = list(matrices)
        if len(matrices) == 0:
            return None
        if len(matrices) == 1:
            return matrices[0]
        all_wires = Wires.all_wires([m.wires for m in matrices], sort=True)
        all_wires = cls.make_wires_continuous(all_wires)
        batch_sizes = [m.batch_size for m in matrices if m.batch_size is not None] + [None]
        batch_size = batch_sizes[0]
        if batch_size is None:
            matrix = pnp.eye(2 * len(all_wires), dtype=complex)
        else:
            matrix = pnp.zeros((batch_size, 2 * len(all_wires), 2 * len(all_wires)), dtype=complex)
            matrix[:, ...] = pnp.eye(2 * len(all_wires), dtype=matrix.dtype)

        matrix = utils.math.convert_and_cast_tensor_from_tensors(
            matrix,
            [m.matrix() for m in matrices],
            cast_priorities=cls.casting_priorities,
        )
        seen_wires = set()
        for m in matrices:
            if m.sorted_wires in seen_wires:
                raise ValueError(f"Cannot have repeated wires in the matrices: {m.sorted_wires}")
            wire0_idx = all_wires.index(m.sorted_wires[0])
            slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + m.shape[-2])
            slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + m.shape[-1])
            # matrix[..., slice_0, slice_1] = qml.math.einsum(
            #     "...ij,...jk->...ik",
            #     detach(matrix[..., slice_0, slice_1]),
            #     utils.math.convert_and_cast_like(m.matrix(), matrix)
            # )
            matrix[..., slice_0, slice_1] = utils.math.convert_and_cast_like(m.matrix(), matrix)
            seen_wires.update(m.sorted_wires)
        return cls(matrix, wires=all_wires)

    def __init__(self, matrix: TensorLike, wires: Wires, **kwargs):
        self._matrix = matrix
        self._wires = wires

    @property
    def wires(self):
        """Wires that the operator acts on.

        Returns:
            Wires: wires
        """
        return self._wires

    @property
    def shape(self):
        return qml.math.shape(self._matrix)

    @property
    def batch_size(self):
        if qml.math.ndim(self._matrix) > 2:
            return self.shape[0]
        return None

    def __array__(self):
        return self.matrix()

    def __matmul__(self, other):
        from ..matchgate_operation import MatchgateOperation

        if isinstance(other, MatchgateOperation):
            other = self.from_operation(other)

        if not isinstance(other, _SingleParticleTransitionMatrix):
            raise ValueError(f"Cannot multiply _SingleTransitionMatrix with {type(other)}")

        all_wires = Wires.all_wires([self.wires, other.wires], sort=True)
        wires = self.make_wires_continuous(all_wires)

        _self = self.pad(wires)
        other = other.pad(wires)

        return _SingleParticleTransitionMatrix(
            qml.math.einsum("...ij,...jk->...ik", _self.matrix(), other.matrix()),
            wires=wires,
        )

    def pad(self, wires: Wires):
        if not isinstance(wires, Wires):
            wires = Wires(wires)
        if self.wires == wires:
            return self
        matrix = self._matrix
        if qml.math.ndim(matrix) == 2:
            padded_matrix = np.eye(2 * len(wires))
        elif qml.math.ndim(matrix) == 3:
            padded_matrix = np.zeros((qml.math.shape(matrix)[0], 2 * len(wires), 2 * len(wires)))
            padded_matrix[:, ...] = np.eye(2 * len(wires))
        else:
            raise NotImplementedError("This method is not implemented yet.")
        padded_matrix = utils.math.convert_and_cast_like(padded_matrix, matrix)
        wire0_idx = wires.index(self.wires[0])
        slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-2])
        slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-1])
        padded_matrix[..., slice_0, slice_1] = matrix
        return _SingleParticleTransitionMatrix(padded_matrix, wires=wires)

    def matrix(self):
        return self._matrix

    @property
    def sorted_wires(self):
        return Wires(sorted(self.wires.tolist()))


class SingleParticleTransitionMatrixOperation(_SingleParticleTransitionMatrix, Operation):
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
            return angles

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
        cls, op: Union[Any, "SingleParticleTransitionMatrixOperation"], **kwargs
    ) -> "SingleParticleTransitionMatrixOperation":
        if isinstance(op, SingleParticleTransitionMatrixOperation):
            return op
        if hasattr(op, "single_particle_transition_matrix"):
            return SingleParticleTransitionMatrixOperation(
                op.single_particle_transition_matrix, wires=op.wires, **kwargs
            )
        if hasattr(op, "to_sptm_operation") and callable(op.to_sptm_operation):
            return op.to_sptm_operation()
        raise ValueError(
            f"Cannot convert {type(op)} to {cls.__name__} "
            f"without the attribute 'single_particle_transition_matrix' or the method 'to_sptm_operation'."
        )

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
            # matrix[..., slice_0, slice_1] = fermionic_operator_matmul(
            #     first_matrix=detach(matrix[..., slice_0, slice_1]),
            #     second_matrix=utils.math.convert_and_cast_like(op_matrix, matrix)
            # )
        # kwargs.setdefault("normalize", True)
        return SingleParticleTransitionMatrixOperation(matrix, wires=all_wires, **kwargs)

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        wires = kwargs.get("wires", None)
        assert wires is not None, "wires kwarg must be set."
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.normal(size=(([batch_size] if batch_size is not None else []) + [2 * len(wires), 2 * len(wires)]))

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(
            cls.random_params(batch_size=batch_size, wires=wires, **kwargs),
            wires=wires,
            **kwargs,
        )

    def __init__(
        self,
        matrix,
        wires: Optional[Union[Sequence[int], Wires]] = None,
        *,
        id=None,
        clip_angles: bool = DEFAULT_CLIP_ANGLES,
        check_angles: bool = DEFAULT_CHECK_ANGLES,
        check_matrix: bool = DEFAULT_CHECK_MATRIX,
        normalize: bool = DEFAULT_NORMALIZE,
        **kwargs,
    ):
        if normalize:
            matrix = orthonormalize(matrix)
        _SingleParticleTransitionMatrix.__init__(self, matrix, wires=wires)
        if self.batch_size is None:
            params = matrix.reshape(-1)
        else:
            params = matrix.reshape(self.batch_size, -1)
        Operation.__init__(self, params, wires=wires, id=id)
        self._hyperparameters = {
            "clip_angles": clip_angles,
            "check_angles": check_angles,
            "check_matrix": check_matrix,
        }
        if check_matrix:
            if not self.check_is_in_so4():
                raise ValueError(f"Matrix is not in SO(4): {matrix}")

    @property
    def sorted_wires(self):
        return Wires(sorted(self.wires.tolist()))

    @property
    def cs_wires(self):
        return Wires(make_wires_continuous(self.wires))

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
        if not isinstance(other, _SingleParticleTransitionMatrix):
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
