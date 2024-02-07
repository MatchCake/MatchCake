import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires
from .fermionic_controlled_z import fCZ
from .fermionic_hadamard import fH
from .matchgate_operation import MatchgateOperation
from .. import matchgate_parameter_sets as mps


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


class _FermionicCNOT(MatchgateOperation):
    num_wires = 2
    num_params = 0

    def __init__(
            self,
            wires=None,
            id=None,
            **kwargs
    ):
        in_params = mps.MatchgatePolarParams.parse_from_params(mps.fCNOT, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, **kwargs)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name


fCNOT = FermionicCNOT
fCNOT.__name__ = "fCNOT"
fCNOT.__doc__ = FermionicCNOT.__doc__

FastfCNOT = _FermionicCNOT
FastfCNOT.__name__ = "FastfCNOT"
FastfCNOT.__doc__ = _FermionicCNOT.__doc__
