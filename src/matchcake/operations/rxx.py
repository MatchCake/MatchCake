from typing import Union

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as pnp

from ..base.matchgate import Matchgate
from .. import matchgate_parameter_sets as mps
from .. import utils

from .matchgate_operation import MatchgateOperation


class Rxx(MatchgateOperation):
    r"""
    This operation implements the following as a MatchgateOperation:

    .. math::
        U = e^{-i\theta X_{j} \otimes X_{j+1}} \\
         = \begin{bmatrix}
            \cos(\theta) & 0 & 0 & -i\sin(\theta) \\
            0 & \cos(\theta) & -i\sin(\theta) & 0 \\
            0 & -i\sin(\theta) & \cos(\theta) & 0 \\
            -i\sin(\theta) & 0 & 0 & \cos(\theta)
        \end{bmatrix}

    where :math:`\theta` is a parameter, :math:`X_{j}` is the Pauli-X operator applied on the wire :math:`j`,
    and :math:`i` is the imaginary unit.
    """
    num_wires = 2
    num_params = 1

    def __init__(
            self,
            params: Union[pnp.ndarray, list, tuple],
            wires=None,
            id=None,
            **kwargs
    ):
        shape = qml.math.shape(params)[-1:]
        n_params = shape[0] if shape else 1
        if n_params != self.num_params:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.num_params} parameters; got {n_params}."
            )
        self._given_params = params
        theta = params
        m_params = mps.MatchgateStandardParams(
            a=qml.math.cos(theta),
            b=-1j * qml.math.sin(theta),
            w=qml.math.cos(theta),
            x=-1j * qml.math.sin(theta),
            y=-1j * qml.math.sin(theta),
            z=qml.math.cos(theta),
            c=-1j * qml.math.sin(theta),
            d=qml.math.cos(theta),
        )
        kwargs["in_param_type"] = mps.MatchgateStandardParams
        super().__init__(m_params, wires=wires, id=id, **kwargs)
