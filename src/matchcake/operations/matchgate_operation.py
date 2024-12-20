from typing import Union

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp
from pennylane.wires import Wires

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps, utils
from .single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation
)


class MatchgateOperation(Matchgate, Operation):
    num_params = mps.MatchgatePolarParams.N_PARAMS
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    casting_priorities = ["numpy", "autograd", "jax", "tf", "torch"]  # greater index means higher priority

    @staticmethod
    def _matrix(*params):
        # TODO: maybe remove this method to use only compute_matrix
        polar_params = mps.MatchgatePolarParams(*params)
        std_params = mps.MatchgateStandardParams.parse_from_params(polar_params)
        matrix = std_params.to_matrix()
        if qml.math.get_interface(matrix) == "torch":
            matrix = matrix.resolve_conj()
        return matrix
    
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
        np_params = self.polar_params.to_vector()
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
        if batch_size in [0, ]:
            return None
        return batch_size

    def get_padded_single_particle_transition_matrix(self, wires=None):
        r"""
        Return the padded single particle transition matrix in order to have the block diagonal form where
        the block is the single particle transition matrix at the corresponding wires.
        
        :param wires: The wires of the whole system.
        :return: padded single particle transition matrix
        """
        if wires is None:
            wires = self.wires
        wires = Wires(wires)
        matrix = self.single_particle_transition_matrix
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
        
        # wire1_idx = wires.index(self.wires[1])
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
        new_std_params = self.standard_params.adjoint()
        new_polar_params = mps.MatchgatePolarParams.parse_from_params(new_std_params)
        return MatchgateOperation(
            new_polar_params,
            wires=self.wires,
            in_param_type=mps.MatchgatePolarParams,
        )
    
    def __matmul__(self, other):
        if isinstance(other, SingleParticleTransitionMatrixOperation):
            return SingleParticleTransitionMatrixOperation.from_operation(self) @ other

        if not isinstance(other, MatchgateOperation):
            raise ValueError(f"Cannot multiply MatchgateOperation with {type(other)}")
        
        if self.wires != other.wires:
            raise NotImplementedError("Cannot multiply MatchgateOperation with different wires yet.")

        new_std_params = mps.MatchgateStandardParams.from_matrix(
            qml.math.einsum(
                "...ij,...jk->...ik",
                self.standard_params.to_matrix(),
                other.standard_params.to_matrix()
            )
        )
        new_polar_params = mps.MatchgatePolarParams.parse_from_params(new_std_params, force_cast_to_real=True)
        return MatchgateOperation(
            new_polar_params,
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

    def to_sptm_operation(self):
        return SingleParticleTransitionMatrixOperation(
            self.single_particle_transition_matrix,
            wires=self.wires
        )



