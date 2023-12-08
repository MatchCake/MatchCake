from typing import Union

import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps

from .matchgate_operation import MatchgateOperation


class MatchgateRotation(Operation):
    num_wires = 2
    num_params = 1
    ndim_params = (1,)

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
        if n_params != 2:
            raise ValueError(
                f"{self.__class__.__name__} requires 2 parameters; got {n_params}."
            )
        m_params = mps.MatchgatePolarParams(r0=1, r1=1, theta0=params[0], theta2=params[1])
        super().__init__(self, m_params, wires=wires, id=id, backend=backend, **kwargs)


MRot = MatchgateRotation

