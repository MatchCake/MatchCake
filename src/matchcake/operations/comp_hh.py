import numpy as np
from pennylane import numpy as pnp
from pennylane.wires import Wires

from .. import matchgate_parameter_sets as mps
from .matchgate_operation import MatchgateOperation


class CompHH(MatchgateOperation):
    num_wires = 2
    num_params = 0

    @classmethod
    def random(cls, wires: Wires, batch_size=None, **kwargs):
        return cls(wires=wires, **kwargs)

    def __init__(self, wires=None, id=None, **kwargs):
        inv_sqrt_2 = 1 / np.sqrt(2)
        m_params = mps.MatchgateStandardParams(
            a=inv_sqrt_2,
            b=inv_sqrt_2,
            c=inv_sqrt_2,
            d=-inv_sqrt_2,
            w=inv_sqrt_2,
            x=inv_sqrt_2,
            y=inv_sqrt_2,
            z=-inv_sqrt_2,
        )
        in_params = mps.MatchgatePolarParams.parse_from_params(m_params, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, **kwargs)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or self.name
