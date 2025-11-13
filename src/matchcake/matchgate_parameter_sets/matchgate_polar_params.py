from dataclasses import dataclass
from typing import Optional, List

import torch
import pennylane as qml
from .matchgate_params import MatchgateParams
from .matchgate_standard_params import MatchgateStandardParams
from ..typing import TensorLike
from ..utils.torch_utils import to_tensor


@dataclass
class MatchgatePolarParams(MatchgateParams):
    r"""
    Matchgate from polar parameters.

    They are the parameters of a Matchgate operation in the standard form which is a 4x4 matrix

    .. math::

            \begin{bmatrix}
                r_0 e^{i\theta_0} & 0 & 0 & (\sqrt{1 - r_0^2}) e^{-i(\theta_1+\pi)} \\
                0 & r_1 e^{i\theta_2} & (\sqrt{1 - r_1^2}) e^{-i(\theta_3+\pi)} & 0 \\
                0 & (\sqrt{1 - r_1^2}) e^{i\theta_3} & r_1 e^{-i\theta_2} & 0 \\
                (\sqrt{1 - r_0^2}) e^{i\theta_1} & 0 & 0 & r_0 e^{-i\theta_0}
            \end{bmatrix}

        where :math:`r_0, r_1, \theta_0, \theta_1, \theta_2, \theta_3, \theta_4` are the parameters.

    The polar parameters will be converted to standard parameterization. The conversion is given by

    .. math::
        \begin{align}
            a &= r_0 e^{i\theta_0} \\
            b &= (\sqrt{1 - r_0^2}) e^{i(\theta_2+\theta_4-(\theta_1+\pi))} \\
            c &= (\sqrt{1 - r_0^2}) e^{i\theta_1} \\
            d &= r_0 e^{i(\theta_2+\theta_4-\theta_0)} \\
            w &= r_1 e^{i\theta_2} \\
            x &= (\sqrt{1 - r_1^2}) e^{i(\theta_2+\theta_4-(\theta_3+\pi))} \\
            y &= (\sqrt{1 - r_1^2}) e^{i\theta_3} \\
            z &= r_1 e^{i\theta_4}
        \end{align}
    """
    r0: Optional[TensorLike] = None
    r1: Optional[TensorLike] = None
    theta0: Optional[TensorLike] = None
    theta1: Optional[TensorLike] = None
    theta2: Optional[TensorLike] = None
    theta3: Optional[TensorLike] = None
    theta4: Optional[TensorLike] = None

    def __post_init__(self):
        if self.theta4 is None and self.theta2 is not None:
            self.theta4 = -self.theta2

    def get_params_list(self) -> List[Optional[TensorLike]]:
        return [self.r0, self.r1, self.theta0, self.theta1, self.theta2, self.theta3, self.theta4]

    def get_modules_params_list(self) -> List[Optional[TensorLike]]:
        return [self.r0, self.r1]

    def get_angles_params_list(self) -> List[Optional[TensorLike]]:
        return [self.theta0, self.theta1, self.theta2, self.theta3, self.theta4]

    def matrix(
            self,
            dtype: torch.dtype = torch.complex128,
            device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        division_epsilon = 1e-12

        shapes = [qml.math.shape(p) for p in self.get_params_list() if p is not None]
        batch_sizes = list(set([s[0] for s in shapes if len(s) > 0]))
        assert len(batch_sizes) <= 1, f"Expect the same batch size for every parameters. Got: {batch_sizes}."
        batch_size = batch_sizes[0] if len(batch_sizes) > 0 else 1
        r0, r1 = [
            (
                to_tensor(p, dtype=dtype, device=device)
                if p is not None
                else torch.ones((batch_size,), dtype=dtype, device=device)
            )
            for p in self.get_modules_params_list()
        ]
        theta0, theta1, theta2, theta3, theta4 = [
            (
                to_tensor(p, dtype=dtype, device=device)
                if p is not None
                else torch.zeros((batch_size,), dtype=dtype, device=device)
            )
            for p in self.get_angles_params_list()
        ]

        r0_tilde = torch.sqrt(1 - r0 ** 2 + division_epsilon)
        r1_tilde = torch.sqrt(1 - r1 ** 2 + division_epsilon)
        return MatchgateStandardParams(
            a=r0 * torch.exp(1j * theta0),
            b=r0_tilde * torch.exp(1j * (theta2 + theta4 - (theta1 + torch.pi))),
            c=r0_tilde * torch.exp(1j * theta1),
            d=r0 * torch.exp(1j * (theta2 + theta4 - theta0)),
            w=r1 * torch.exp(1j * theta2),
            x=r1_tilde * torch.exp(1j * (theta2 + theta4 - (theta3 + torch.pi))),
            y=r1_tilde * torch.exp(1j * theta3),
            z=r1 * torch.exp(1j * theta4),
        ).matrix(dtype=dtype, device=device)




