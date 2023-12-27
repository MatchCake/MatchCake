import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires
from .fermionic_controlled_z import fCZ
from .fermionic_hadamard import fH


class FermionicCNOT(Operation):
    num_wires = 2
    num_params = 0

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        wires = Wires(wires)
        # TODO: must be closer than (I ⊗ H) CZ (I ⊗ H)
        return [fH(wires=wires), fCZ(wires=wires), fH(wires=wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name


fCNOT = FermionicCNOT
fCNOT.__name__ = "fCNOT"
