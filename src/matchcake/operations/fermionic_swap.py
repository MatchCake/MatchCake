import numpy as np
from pennylane import numpy as pnp
from pennylane.wires import Wires

from .. import matchgate_parameter_sets as mps
from .matchgate_operation import MatchgateOperation


class CompZX(MatchgateOperation):
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, *, backend=pnp, **kwargs):
        in_params = mps.MatchgatePolarParams.parse_from_params(mps.fSWAP, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, backend=backend, **kwargs)


FermionicSWAP = CompZX
FermionicSWAP.__name__ = "FermionicSWAP"

fSWAP = FermionicSWAP
fSWAP.__name__ = "fSWAP"


def fswap_chain_gen(wires, **kwargs):
    is_reverse = wires[0] > wires[1]
    wire0, wire1 = list(sorted(wires))
    wires_gen = range(wire0, wire1) if is_reverse else reversed(range(wire0, wire1))
    for tmp_wire0 in wires_gen:
        yield fSWAP(wires=[tmp_wire0, tmp_wire0 + 1], **kwargs)
    return


def fswap_chain(wires, **kwargs):
    return [op for op in fswap_chain_gen(wires, **kwargs)]
