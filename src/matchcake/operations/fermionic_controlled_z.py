from typing import Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.operation import Operation

from .. import matchgate_parameter_sets as mps
from .. import utils
from ..base.matchgate import Matchgate
from .fermionic_hadamard import fH
from .fermionic_paulis import fXX
from .fermionic_swap import fSWAP
from .matchgate_operation import MatchgateOperation


class FermionicControlledZ(Operation):
    num_wires = 2
    num_params = 0

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = qml.wires.Wires(wires)
        return [
            fH(wires=wires),
            fSWAP(wires=wires),
            fXX(wires=wires),
            fH(wires=wires),
        ]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name


fCZ = FermionicControlledZ
fCZ.__name__ = "fCZ"
