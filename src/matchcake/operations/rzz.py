from typing import Optional

import pennylane as qml
import torch
from pennylane.typing import TensorLike

from .matchgate_operation import MatchgateOperation
from .. import matchgate_parameter_sets as mgp


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

    def __init__(
            self,
            theta: TensorLike,
            wires=None,
            id=None,
            default_dtype: torch.dtype = torch.complex128,
            default_device: Optional[torch.device] = None,
            **kwargs,
    ):
        super().__init__(
            mgp.MatchgateStandardParams(
                a=qml.math.exp(-1j * theta / 2),
                w=qml.math.exp(1j * theta / 2),
                z=qml.math.exp(1j * theta / 2),
                d=qml.math.exp(-1j * theta / 2),
            ),
            wires=wires,
            id=id,
            default_dtype=default_dtype,
            default_device=default_device,
            **kwargs
        )

    # TODO: Add constraints to the angle as n*pi/2
