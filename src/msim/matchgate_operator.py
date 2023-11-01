from typing import Union

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from .matchgate import Matchgate
from . import matchgate_parameter_sets as mps


class MatchgateOperator(Matchgate, Operation):
    num_params = 6
    num_wires = 2
    par_domain = "A"

    grad_method = "A"
    grad_recipe = None

    generator = None

    @staticmethod
    def _matrix(*params):
        polar_params = mps.MatchgatePolarParams(*params)
        std_params = mps.MatchgateStandardParams.parse_from_params(polar_params)
        return pnp.array(std_params.to_matrix())
    
    def __init__(
            self,
            params: Union[mps.MatchgateParams, pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            *,
            backend=pnp
    ):
        Matchgate.__init__(self, params, backend=backend)
        Operation.__init__(self, *self.polar_params.to_numpy(), wires=wires, id=id)

    def adjoint(self):
        return MatchgateOperator(self.standard_params.adjoint(), wires=self.wires, backend=self.backend)
