from typing import Union

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from .matchgate import Matchgate
from . import matchgate_parameter_sets as mps


class MatchgateOperator(Matchgate, Operation):
    num_params = 7
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
        Matchgate.__init__(self, params, backend=backend, **kwargs)
        params = self.composed_hamiltonian_params.to_numpy()
        self.num_params = len(params)
        Operation.__init__(self, *params, wires=wires, id=id)

    def adjoint(self):
        return MatchgateOperator(self.standard_params.adjoint(), wires=self.wires, backend=self.backend)
