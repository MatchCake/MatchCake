import pennylane as qml
from pennylane.operation import Operation

from .comp_hh import CompHH
from .comp_paulis import CompXX
from .fermionic_swap import fSWAP


class FermionicControlledZ(Operation):
    num_wires = 2
    num_params = 0

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = qml.wires.Wires(wires)
        return [
            CompHH(wires=wires),
            fSWAP(wires=wires),
            CompXX(wires=wires),
            CompHH(wires=wires),
        ]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name


fCZ = FermionicControlledZ
fCZ.__name__ = "fCZ"
