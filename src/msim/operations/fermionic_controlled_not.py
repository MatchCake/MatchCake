from typing import Union

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps
from .. import utils

from .matchgate_operation import MatchgateOperation
from .fermionic_hadamard import fH


class FermionicCNOT(Operation):
    num_wires = 2
    num_params = 0
    
    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = qml.wires.Wires(wires)
        raise NotImplementedError("Decomposition of FermionicCNOT is not implemented yet.")


fCNOT = FermionicCNOT
fCNOT.__name__ = "fCNOT"
