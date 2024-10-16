from typing import Union, Iterable

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp
from pennylane.wires import Wires

from ...base.matchgate import Matchgate
from ... import matchgate_parameter_sets as mps, utils
from ...templates import TensorLike
from ..matchgate_operation import MatchgateOperation
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
            op: Union[MatchgateOperation, "_SingleParticleTransitionMatrix"]
    ) -> "_SingleParticleTransitionMatrix":
        if isinstance(op, _SingleParticleTransitionMatrix):
            return op
        return cls(op.single_particle_transition_matrix, op.wires)

    @classmethod
    def from_operations(
            cls,
            ops: Iterable[Union[MatchgateOperation, "_SingleParticleTransitionMatrix"]]
    ) -> "_SingleParticleTransitionMatrix":
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
        return cls(matrix, all_wires)

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
        return cls(matrix, all_wires)

    def __init__(self, matrix: TensorLike, wires: Wires):
        self._matrix = matrix
        self.wires = wires

    @property
    def shape(self):
        return qml.math.shape(self.matrix())

    @property
    def batch_size(self):
        if qml.math.ndim(self.matrix()) > 2:
            return self.shape[0]
        return None

    def __array__(self):
        return self.matrix()

    def __matmul__(self, other):
        if isinstance(other, MatchgateOperation):
            other = _SingleParticleTransitionMatrix.from_operation(other)

        if not isinstance(other, _SingleParticleTransitionMatrix):
            raise ValueError(f"Cannot multiply _SingleTransitionMatrix with {type(other)}")

        if self.wires == other.wires:
            wires = self.wires
        else:
            all_wires = Wires.all_wires([self.wires, other.wires], sort=True)
            wires = self.make_wires_continuous(all_wires)

        _self = self.pad(wires)
        other = other.pad(wires)

        return self.__class__(
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
        matrix = self.matrix()
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
        return self.__class__(padded_matrix, wires)

    def matrix(self):
        return self._matrix


class SingleParticleTransitionMatrixOperation(Operation, _SingleParticleTransitionMatrix):
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority

    def __init__(self, matrix, wires=None, id=None, **kwargs):
        self._num_params = np.prod(qml.math.shape(matrix)[-2:])
        _SingleParticleTransitionMatrix.__init__(self, matrix, wires)
        if self.batch_size is None:
            params = matrix.reshape(-1)
        else:
            params = matrix.reshape(self.batch_size, -1)
        Operation.__init__(self, params, wires, id, **kwargs)

    def matrix(self, wire_order=None):
        if self.wires != Wires(wire_order):
            raise ValueError(f""
                             f"Invalid wire order: {wire_order}. "
                             f"Expected: {self.wires}. "
                             f"We currently cannot permute the wires with {self.__class__.__name__}.")
        return self._matrix



class SptmRxRx(SingleParticleTransitionMatrixOperation):
    num_params = 2

    def __init__(self, params, wires=None, id=None, **kwargs):
        params_shape = qml.math.shape(params)
        if len(params_shape) == 1:
            matrix = np.zeros((4, 4), dtype=complex)
        elif len(params_shape) == 2:
            matrix = np.zeros((params_shape[0], 4, 4), dtype=complex)
        else:
            raise ValueError(f"Invalid shape for the parameters: {params_shape}")

        if params_shape[-1] != 2:
            raise ValueError(f"Invalid number of parameters: {params_shape[-1]}. Expected 2.")

        matrix = convert_and_cast_like(matrix, params)
        theta, phi = params[..., 0], params[..., 1]

        matrix[..., 0, 0] = qml.math.cos(phi/2 - theta/2)
        matrix[..., 0, 3] = -qml.math.sin(phi/2 - theta/2)
        matrix[..., 1, 1] = qml.math.cos(phi/2 + theta/2)
        matrix[..., 1, 2] = qml.math.sin(phi/2 + theta/2)
        matrix[..., 2, 1] = -qml.math.sin(phi/2 + theta/2)
        matrix[..., 2, 2] = qml.math.cos(phi/2 + theta/2)
        matrix[..., 3, 0] = qml.math.sin(phi/2 - theta/2)
        matrix[..., 3, 3] = qml.math.cos(phi/2 - theta/2)
        super().__init__(matrix, wires, id, **kwargs)


