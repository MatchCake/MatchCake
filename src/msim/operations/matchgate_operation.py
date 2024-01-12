from typing import Union, Iterable

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp
from pennylane.wires import Wires

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps, utils


class MatchgateOperation(Matchgate, Operation):
    num_params = mps.MatchgatePolarParams.N_PARAMS
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    @staticmethod
    def _matrix(*params):
        # TODO: maybe remove this method to use only compute_matrix
        polar_params = mps.MatchgatePolarParams(*params, backend=pnp)
        std_params = mps.MatchgateStandardParams.parse_from_params(polar_params)
        return pnp.array(std_params.to_matrix())
    
    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return MatchgateOperation._matrix(*params)
    
    def __init__(
            self,
            params: Union[mps.MatchgateParams, pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            **kwargs
    ):
        in_param_type = kwargs.get("in_param_type", mps.MatchgatePolarParams)
        in_params = in_param_type.parse_from_any(params)
        Matchgate.__init__(self, in_params, **kwargs)
        np_params = self.polar_params.to_numpy()
        self.num_params = len(np_params)
        self.draw_label_params = kwargs.get("draw_label_params", None)
        Operation.__init__(self, *np_params, wires=wires, id=id)

    @property
    def batch_size(self):
        not_none_params = [
            p for p in
            self.get_all_params_set(make_params=False)
            if p is not None
        ]
        if len(not_none_params) == 0:
            raise ValueError("No params set. Cannot make standard params.")
        batch_size = not_none_params[0].batch_size
        if batch_size in [0, 1]:
            return None
        return batch_size

    def get_padded_single_transition_particle_matrix(self, wires=None):
        r"""
        Return the padded single transition particle matrix in order to have the block diagonal form where
        the block is the single transition particle matrix at the corresponding wires.
        
        :param wires: The wires of the whole system.
        :return: padded single transition particle matrix
        """
        if wires is None:
            wires = self.wires
        matrix = self.single_transition_particle_matrix
        if qml.math.ndim(matrix) == 2:
            padded_matrix = pnp.eye(2*len(wires))
        elif qml.math.ndim(matrix) == 3:
            padded_matrix = pnp.zeros((qml.math.shape(matrix)[0], 2*len(wires), 2*len(wires)))
            padded_matrix[:, ...] = pnp.eye(2*len(wires))
        else:
            raise ValueError(f"Cannot pad matrix of ndim {qml.math.ndim(matrix)}.")
        padded_matrix = utils.math.convert_and_cast_like(padded_matrix, matrix)
        wire0_idx = wires.index(self.wires[0])
        # wire0_submatrix = matrix[:matrix.shape[0]//2, :matrix.shape[1]//2]
        # wire0_shape = wire0_submatrix.shape
        # wire0_slice0 = slice(2 * wire0_idx, 2 * wire0_idx + wire0_shape[0])
        # wire0_slice1 = slice(2 * wire0_idx, 2 * wire0_idx + wire0_shape[1])
        
        wire1_idx = wires.index(self.wires[1])
        # wire1_submatrix = matrix[matrix.shape[0]//2:, matrix.shape[1]//2:]
        # wire1_shape = wire1_submatrix.shape
        # wire1_slice0 = slice(2 * wire1_idx, 2 * wire1_idx + wire1_shape[0])
        # wire1_slice1 = slice(2 * wire1_idx, 2 * wire1_idx + wire1_shape[1])
        
        # padded_matrix[wire0_slice0, wire0_slice1] = wire0_submatrix
        # padded_matrix[wire1_slice0, wire1_slice1] = wire1_submatrix
        slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-2])
        slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-1])
        padded_matrix[..., slice_0, slice_1] = matrix
        return padded_matrix
    
    def adjoint(self):
        return MatchgateOperation(
            self.polar_params.adjoint(),
            wires=self.wires,
            in_param_type=mps.MatchgatePolarParams,
        )
    
    def __matmul__(self, other):
        if not isinstance(other, MatchgateOperation):
            raise ValueError(f"Cannot multiply MatchgateOperation with {type(other)}")
        
        if self.wires != other.wires:
            raise NotImplementedError("Cannot multiply MatchgateOperation with different wires yet.")
        
        std_params = mps.MatchgateStandardParams.from_matrix(self.matrix() @ other.matrix())
        polar_params = mps.MatchgatePolarParams.parse_from_params(std_params)
        return MatchgateOperation(
            polar_params,
            wires=self.wires,
            in_param_type=mps.MatchgatePolarParams,
        )
    
    def label(self, decimals=None, base_label=None, cache=None):
        if self.draw_label_params is None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)

        op_label = base_label or self.__class__.__name__
        return f"{op_label}({self.draw_label_params})"

    def __repr__(self):
        return Operation.__repr__(self)

    def __str__(self):
        return Operation.__str__(self)

    def __copy__(self):
        return Operation.__copy__(self)


class _SingleTransitionMatrix:

    @staticmethod
    def make_wires_continuous(wires: Wires):
        wires_array = wires.tolist()
        min_wire, max_wire = min(wires_array), max(wires_array)
        return Wires(range(min_wire, max_wire + 1))

    @classmethod
    def from_operation(cls, op: MatchgateOperation):
        return cls(op.single_transition_particle_matrix, op.wires)

    @classmethod
    def from_operations(cls, ops: Iterable[MatchgateOperation]):
        if len(ops) == 0:
            return None
        if len(ops) == 1:
            return cls.from_operation(next(iter(ops)))
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
            slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + op.single_transition_particle_matrix.shape[-2])
            slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + op.single_transition_particle_matrix.shape[-1])
            matrix[..., slice_0, slice_1] = op.single_transition_particle_matrix
        return cls(matrix, all_wires)

    def __init__(self, matrix: pnp.ndarray, wires: Wires):
        self.matrix = matrix
        self.wires = wires

    def __array__(self):
        return self.matrix

    def __matmul__(self, other):
        if not isinstance(other, _SingleTransitionMatrix):
            raise ValueError(f"Cannot multiply _SingleTransitionMatrix with {type(other)}")

        if self.wires != other.wires:
            raise NotImplementedError("Cannot multiply _SingleTransitionMatrix with different wires yet.")

        return _SingleTransitionMatrix(self.matrix @ other.matrix, self.wires)

    def pad(self, wires: Wires):
        if self.wires == wires:
            return self
        matrix = self.matrix
        if qml.math.ndim(matrix) == 2:
            padded_matrix = pnp.eye(2 * len(wires), dtype=matrix.dtype)
        elif qml.math.ndim(matrix) == 3:
            padded_matrix = pnp.zeros((qml.math.shape(matrix)[0], 2 * len(wires), 2 * len(wires)), dtype=matrix.dtype)
            padded_matrix[:, ...] = pnp.eye(2 * len(wires), dtype=matrix.dtype)
        else:
            raise NotImplementedError("This method is not implemented yet.")

        wire0_idx = wires.index(self.wires[0])
        # wire1_idx = wires.index(self.wires[1])
        slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-2])
        slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[-1])
        padded_matrix[..., slice_0, slice_1] = matrix
        return _SingleTransitionMatrix(padded_matrix, wires)
