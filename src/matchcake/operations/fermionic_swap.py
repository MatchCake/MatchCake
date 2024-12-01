import numpy as np
from pennylane import numpy as pnp
from pennylane.wires import Wires

from .matchgate_operation import MatchgateOperation
from .. import matchgate_parameter_sets as mps


class FermionicSWAP(MatchgateOperation):
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)
    
    def __init__(
            self,
            wires=None,
            id=None,
            *,
            backend=pnp,
            **kwargs
    ):
        in_params = mps.MatchgatePolarParams.parse_from_params(mps.fSWAP, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, backend=backend, **kwargs)


fSWAP = FermionicSWAP
fSWAP.__name__ = "fSWAP"
