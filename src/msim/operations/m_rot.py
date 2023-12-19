from typing import Union

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps

from .matchgate_operation import MatchgateOperation


class MatchgateRotation(MatchgateOperation):
    num_wires = 2
    num_params = 2

    def __init__(
            self,
            params: Union[pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            *,
            backend=pnp,
            **kwargs
    ):
        shape = qml.math.shape(params)[-1:]
        n_params = shape[0]
        if n_params != self.num_params:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.num_params} parameters; got {n_params}."
            )
        m_params = mps.MatchgateStandardParams(
            a=pnp.cos(params[0] / 2), b=-1j*pnp.sin(params[0] / 2),
            c=-1j*pnp.sin(params[0] / 2), d=pnp.cos(params[0] / 2),
            w=pnp.cos(params[1] / 2), x=-1j*pnp.sin(params[1] / 2),
            y=-1j*pnp.sin(params[1] / 2), z=pnp.cos(params[1] / 2),
        )
        in_params = mps.MatchgatePolarParams.parse_from_params(m_params, force_cast_to_real=True)
        kwargs["in_param_type"] = mps.MatchgatePolarParams
        super().__init__(in_params, wires=wires, id=id, backend=backend, **kwargs)


MRot = MatchgateRotation


def mrot_template(param0, param1, wires):
    MRot([param0, param1], wires=wires)
