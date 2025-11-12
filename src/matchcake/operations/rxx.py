from typing import Union, Optional

import pennylane as qml
import torch
from pennylane.typing import TensorLike

from .. import matchgate_parameter_sets as mps
from .matchgate_operation import MatchgateOperation
from ..utils.torch_utils import to_tensor


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

    def __new__(cls, theta: TensorLike, wires=None, id=None, **kwargs):
        return cls.from_std_params(
            a=qml.math.cos(theta / 2),
            b=-1j * qml.math.sin(theta / 2),
            w=qml.math.cos(theta / 2),
            x=-1j * qml.math.sin(theta / 2),
            y=-1j * qml.math.sin(theta / 2),
            z=qml.math.cos(theta / 2),
            c=-1j * qml.math.sin(theta / 2),
            d=qml.math.cos(theta / 2),
            wires=wires,
            id=id,
            **kwargs
        )

    # def __init__(
    #         self,
    #         params: TensorLike,
    #         wires=None,
    #         id=None,
    #         default_dtype: torch.dtype = torch.complex128,
    #         default_device: Optional[torch.device] = None,
    #         **kwargs,
    # ):
    #     shape = qml.math.shape(params)[-1:]
    #     n_params = shape[0] if shape else 1
    #     if n_params != self.num_params:
    #         raise ValueError(f"{self.__class__.__name__} requires {self.num_params} parameters; got {n_params}.")
    #     self._given_params = params
    #     if qml.math.get_interface(params) != "torch":
    #         theta = to_tensor(params, dtype=default_dtype, device=default_device)
    #     else:
    #         theta = params
    #     m_params = mps.MatchgateStandardParams(
    #         a=qml.math.cos(theta / 2),
    #         b=-1j * qml.math.sin(theta / 2),
    #         w=qml.math.cos(theta / 2),
    #         x=-1j * qml.math.sin(theta / 2),
    #         y=-1j * qml.math.sin(theta / 2),
    #         z=qml.math.cos(theta / 2),
    #         c=-1j * qml.math.sin(theta / 2),
    #         d=qml.math.cos(theta / 2),
    #     )
    #     # kwargs["in_param_type"] = mps.MatchgateStandardParams
    #     batch_size = qml.math.shape(params)[0]
    #     matrix = torch.zeros(batch_size, 4, 4, dtype=theta.dtype, device=theta.device)
    #     matrix[..., 0, 0] = qml.math.cos(theta / 2)
    #     matrix[..., 0, 3] = -1j * qml.math.sin(theta / 2)
    #     matrix[..., 3, 0] = -1j * qml.math.sin(theta / 2)
    #     matrix[..., 3, 3] = qml.math.cos(theta / 2)
    #     super().__init__(matrix, wires=wires, id=id, **kwargs)
