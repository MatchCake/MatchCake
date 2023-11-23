from typing import Union

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from .matchgate import Matchgate
from . import matchgate_parameter_sets as mps


class MatchgateOperator(Matchgate, Operation):
    num_params = mps.MatchgatePolarParams.N_PARAMS
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    @staticmethod
    def _matrix(*params):
        polar_params = mps.MatchgatePolarParams(*params, backend=pnp)
        std_params = mps.MatchgateStandardParams.parse_from_params(polar_params)
        return pnp.array(std_params.to_matrix())
    
    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return MatchgateOperator._matrix(*params)
    
    def __init__(
            self,
            params: Union[mps.MatchgateParams, pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            *,
            backend=pnp,
            **kwargs
    ):
        in_param_type = kwargs.get("in_param_type", mps.MatchgatePolarParams)
        in_params = in_param_type.parse_from_any(params)
        Matchgate.__init__(self, in_params, backend=backend, **kwargs)
        np_params = self.polar_params.to_numpy()
        self.num_params = len(np_params)
        Operation.__init__(self, *np_params, wires=wires, id=id)
    
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
        padded_matrix = pnp.eye(2*len(wires), dtype=matrix.dtype)
        
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
        slice_0 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[0])
        slice_1 = slice(2 * wire0_idx, 2 * wire0_idx + matrix.shape[1])
        padded_matrix[slice_0, slice_1] = matrix
        return padded_matrix
    
    def adjoint(self):
        return MatchgateOperator(
            self.polar_params.adjoint(),
            wires=self.wires,
            backend=self.backend,
            in_param_type=mps.MatchgatePolarParams,
        )
