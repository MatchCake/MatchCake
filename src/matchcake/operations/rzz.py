from typing import Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.operation import Operation

from .. import matchgate_parameter_sets as mps
from .. import utils
from ..base.matchgate import Matchgate
from .matchgate_operation import MatchgateOperation


class Rzz(MatchgateOperation):
    r"""
    This operation implements the following as a MatchgateOperation:

    .. math::
        U = e^{-i\theta/2 Z_{j} \otimes Z_{j+1}} \\
         = \begin{bmatrix}
            e^{-i\theta/2} & 0 & 0 & 0 \\
            0 & e^{i\theta/2} & 0 & 0 \\
            0 & 0 & e^{i\theta/2} & 0 \\
            0 & 0 & 0 & e^{-i\theta/2}
        \end{bmatrix}

    where :math:`\theta` is a parameter, :math:`Z_{j}` is the Pauli-Z operator applied on the wire :math:`j`,
    and :math:`i` is the imaginary unit.
    """

    num_wires = 2
    num_params = 1

    # TODO: Add constraints to the angle as n*pi/2

    def __init__(self, params: Union[pnp.ndarray, list, tuple], wires=None, id=None, **kwargs):
        shape = qml.math.shape(params)[-1:]
        n_params = shape[0] if shape else 1
        if n_params != self.num_params:
            raise ValueError(f"{self.__class__.__name__} requires {self.num_params} parameters; got {n_params}.")
        self._given_params = params
        theta = params
        m_params = mps.MatchgateStandardParams(
            a=qml.math.exp(-1j * theta / 2),
            w=qml.math.exp(1j * theta / 2),
            z=qml.math.exp(1j * theta / 2),
            d=qml.math.exp(-1j * theta / 2),
        )
        kwargs["in_param_type"] = mps.MatchgateStandardParams
        super().__init__(m_params, wires=wires, id=id, **kwargs)
