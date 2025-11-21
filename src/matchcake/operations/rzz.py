from typing import Optional

import pennylane as qml
import torch
from pennylane.typing import TensorLike
import numpy as np

from .. import matchgate_parameter_sets as mgp
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
    and :math:`i` is the imaginary unit. Since the determinant constraint needs to be fulfilled, theta needs to be
    :math:`\theta = k\pi` for :math:`k \in \mathbf{Z}`.
    """

    @classmethod
    def random_params(cls, batch_size=None, **kwargs):
        params_shape = ([batch_size] if batch_size is not None else []) + [1]
        seed = kwargs.pop("seed", None)
        rn_gen = np.random.default_rng(seed)
        return np.pi * rn_gen.choice([0, 1], size=params_shape)

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
            **kwargs,
        )

    # TODO: Add constraints to the angle as n*pi/2
