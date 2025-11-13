from typing import Optional

import pennylane as qml
import torch
from pennylane.typing import TensorLike

from .matchgate_operation import MatchgateOperation
from .. import matchgate_parameter_sets as mgp


class Rxx(MatchgateOperation):
    r"""
    This operation implements the following as a MatchgateOperation:

    .. math::
        U = e^{-i\theta/2 X_{j} \otimes X_{j+1}} \\
         = \begin{bmatrix}
            \cos(\theta/2) & 0 & 0 & -i\sin(\theta/2) \\
            0 & \cos(\theta/2) & -i\sin(\theta/2) & 0 \\
            0 & -i\sin(\theta/2) & \cos(\theta/2) & 0 \\
            -i\sin(\theta/2) & 0 & 0 & \cos(\theta/2)
        \end{bmatrix}

    where :math:`\theta` is a parameter, :math:`X_{j}` is the Pauli-X operator applied on the wire :math:`j`,
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
                a=qml.math.cos(theta / 2),
                b=-1j * qml.math.sin(theta / 2),
                w=qml.math.cos(theta / 2),
                x=-1j * qml.math.sin(theta / 2),
                y=-1j * qml.math.sin(theta / 2),
                z=qml.math.cos(theta / 2),
                c=-1j * qml.math.sin(theta / 2),
                d=qml.math.cos(theta / 2),
            ),
            wires=wires,
            id=id,
            default_dtype=default_dtype,
            default_device=default_device,
            **kwargs
        )
