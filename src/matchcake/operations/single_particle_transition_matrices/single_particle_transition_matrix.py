from typing import Union, Iterable, Sequence, Optional

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane import numpy as pnp
from pennylane.wires import Wires

from ...base.matchgate import Matchgate
from ... import matchgate_parameter_sets as mps, utils
from ...templates import TensorLike
from ...utils.math import convert_and_cast_like


class _SingleParticleTransitionMatrix:
    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority

    @staticmethod
    def make_wires_continuous(wires: Wires):
        wires_array = wires.tolist()
        min_wire, max_wire = min(wires_array), max(wires_array)
        return Wires(range(min_wire, max_wire + 1))

    @classmethod
    def from_operation(
            cls,
            op: Union["MatchgateOperation", "_SingleParticleTransitionMatrix"]
    ) -> "_SingleParticleTransitionMatrix":
        if isinstance(op, cls):
            return op
        return cls(op.single_particle_transition_matrix, wires=op.wires)

    @classmethod
    def from_operations(
            cls,
            ops: Iterable[Union["MatchgateOperation", "_SingleParticleTransitionMatrix"]]
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
            matrix[..., slice_0, slice_1] = sptm
        return cls(matrix, wires=all_wires)

    @classmethod
    def from_spt_matrices(
            cls,
            matrices: Iterable["_SingleParticleTransitionMatrix"]
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
            matrix, [m.matrix() for m in matrices],
            cast_priorities=cls.casting_priorities
        )
        seen_wires = set()
        for m in matrices:
            if m.wires in seen_wires:
                raise ValueError(f"Cannot have repeated wires in the matrices: {m.wires}")
            wire0_idx = all_wires.index(m.wires[0])
            slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + m.shape[-2])
            slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + m.shape[-1])
            matrix[..., slice_0, slice_1] = utils.math.convert_and_cast_like(m.matrix(), matrix)
            seen_wires.update(m.wires)
        return cls(matrix, wires=all_wires)

    def __init__(self, matrix: TensorLike, wires: Wires):
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

        if self.wires == other.wires:
            wires = self.wires
        else:
            all_wires = Wires.all_wires([self.wires, other.wires], sort=True)
            wires = self.make_wires_continuous(all_wires)

        _self = self.pad(wires)
        other = other.pad(wires)

        return _SingleParticleTransitionMatrix(
            qml.math.einsum(
                "...ij,...jk->...ik",
                _self.matrix(),
                other.matrix()
            ),
            wires=wires
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


class SingleParticleTransitionMatrixOperation(_SingleParticleTransitionMatrix, Operation):
    num_wires = AnyWires
    num_params = 1
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority
    DEFAULT_CHECK_MATRIX = False

    ALLOWED_ANGLES = None
    DEFAULT_CHECK_ANGLES = False
    DEFAULT_CLIP_ANGLES = True

    @classmethod
    def clip_angles(cls, angles):
        """
        If the ALLOWED_ANGLES is not none, set the angles to the closest allowed angle.
        """
        if cls.ALLOWED_ANGLES is None:
            return angles
        angles = qml.math.where(angles >= 0, angles % (2 * np.pi), angles % (-2 * np.pi))
        allowed_angles_array = convert_and_cast_like(np.array(cls.ALLOWED_ANGLES), angles)
        angles_shape = qml.math.shape(angles)
        angles_flatten = qml.math.reshape(angles, (-1, 1))
        distances = qml.math.abs(angles_flatten - allowed_angles_array)
        angles_flatten = allowed_angles_array[qml.math.argmin(distances, -1)]
        angles = qml.math.reshape(angles_flatten, angles_shape)
        return angles

    @classmethod
    def check_angles(cls, angles):
        """
        If the ALLOWED_ANGLES is not none, check if the angles are in the allowed range.
        """
        if not qml.math.all(qml.math.isin(angles, cls.ALLOWED_ANGLES)):
            raise ValueError(f"Invalid angles: {angles}. Expected: {cls.ALLOWED_ANGLES}")
        return True

    def __init__(
            self,
            matrix,
            wires: Optional[Union[Sequence[int], Wires]] = None,
            *,
            id=None,
            clip_angles: bool = DEFAULT_CLIP_ANGLES,
            check_angles: bool = DEFAULT_CHECK_ANGLES,
            check_matrix: bool = DEFAULT_CHECK_MATRIX,
            **kwargs
    ):
        if check_matrix:
            if not self.check_is_in_so4():
                raise ValueError(f"Matrix is not in SO(4): {matrix}")
        _SingleParticleTransitionMatrix.__init__(self, matrix, wires=wires)
        if self.batch_size is None:
            params = matrix.reshape(-1)
        else:
            params = matrix.reshape(self.batch_size, -1)
        Operation.__init__(self, params, wires=wires, id=id, **kwargs)
        self._hyperparameters = {
            "clip_angles": clip_angles,
            "check_angles": check_angles,
            "check_matrix": check_matrix,
        }

    def matrix(self, wire_order=None) -> TensorLike:
        if wire_order is not None and self.wires != Wires(wire_order):
            raise ValueError(f""
                             f"Invalid wire order: {wire_order}. "
                             f"Expected: {self.wires}. "
                             f"We currently cannot permute the wires with {self.__class__.__name__}.")
        wires = Wires(wire_order or self.wires)
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
        return SingleParticleTransitionMatrixOperation(padded_matrix, wires=wires)

    def __matmul__(self, other):
        from ..matchgate_operation import MatchgateOperation

        if isinstance(other, MatchgateOperation):
            other = self.from_operation(other)

        if not isinstance(other, _SingleParticleTransitionMatrix):
            raise ValueError(f"Cannot multiply {self.__class__.__name__} with {type(other)}")

        if self.wires == other.wires:
            wires = self.wires
        else:
            all_wires = Wires.all_wires([self.wires, other.wires], sort=True)
            wires = self.make_wires_continuous(all_wires)

        _self = self.pad(wires)
        other = other.pad(wires)

        return SingleParticleTransitionMatrixOperation(
            qml.math.einsum(
                "...ij,...jk->...ik",
                _self.matrix(),
                other.matrix()
            ),
            wires=wires
        )

    def adjoint(self) -> "SingleParticleTransitionMatrixOperation":
        return SingleParticleTransitionMatrixOperation(
            qml.math.conj(
                qml.math.einsum(
                    "...ij->...ji",
                    self.matrix()
                )
            ),
            wires=self.wires,
            **self._hyperparameters
        )


