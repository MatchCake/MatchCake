from typing import Optional

import pennylane as qml
import torch
from pennylane.typing import TensorLike
import numpy as np

from .. import matchgate_parameter_sets as mgp
from .matchgate_operation import MatchgateOperation


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

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [1]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return rn_gen.uniform(0, 2 * np.pi, params_shape)

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
            **kwargs,
        )
