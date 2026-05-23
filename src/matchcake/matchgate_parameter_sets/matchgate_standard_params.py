from dataclasses import dataclass
from typing import List, Optional, cast

import pennylane as qml
import torch

from ..typing import TensorLike
from ..utils.torch_utils import to_tensor
from .matchgate_params import MatchgateParams


@dataclass
class MatchgateStandardParams(MatchgateParams):
    r"""
    A matchgate is a matrix of the form

    .. math::
        \begin{pmatrix}
            a & 0 & 0 & b \\
            0 & w & x & 0 \\
            0 & y & z & 0 \\
            c & 0 & 0 & d
        \end{pmatrix}

    where :math:`a, b, c, d, w, x, y, z \in \mathbb{C}`. The matrix M can be decomposed as

    .. math::
        A = \begin{pmatrix}
            a & b \\
            c & d
        \end{pmatrix}

    and

    .. math::
        W = \begin{pmatrix}
            w & x \\
            y & z
        \end{pmatrix}
    """

    a: Optional[TensorLike] = None
    b: Optional[TensorLike] = None
    c: Optional[TensorLike] = None
    d: Optional[TensorLike] = None
    w: Optional[TensorLike] = None
    x: Optional[TensorLike] = None
    y: Optional[TensorLike] = None
    z: Optional[TensorLike] = None

    @classmethod
    def from_sub_matrices(
        cls,
        outer_matrix: TensorLike,
        inner_matrix: TensorLike,
    ):
        om = cast(torch.Tensor, outer_matrix)
        im = cast(torch.Tensor, inner_matrix)
        return cls(
            a=om[..., 0, 0],
            b=om[..., 0, 1],
            c=om[..., 1, 0],
            d=om[..., 1, 1],
            w=im[..., 0, 0],
            x=im[..., 0, 1],
            y=im[..., 1, 0],
            z=im[..., 1, 1],
        )

    def get_params_list(self) -> List[Optional[TensorLike]]:
        return [self.a, self.b, self.c, self.d, self.w, self.x, self.y, self.z]

    def matrix(self, dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> torch.Tensor:
        shapes = [qml.math.shape(p) for p in self.get_params_list() if p is not None]
        batch_sizes = list(set([s[0] for s in shapes if len(s) > 0]))
        assert len(batch_sizes) <= 1, f"Expect the same batch size for every parameters. Got: {batch_sizes}."
        batch_size = self.batch_size if self.batch_size is not None else 1
        a, b, c, d, w, x, y, z = cast(
            tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            ],
            tuple(
                cast(torch.Tensor, to_tensor(p, dtype=dtype, device=device))
                if p is not None
                else torch.zeros((batch_size,), dtype=dtype, device=device)
                for p in self.get_params_list()
            ),
        )
        matrix = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)
        matrix[..., 0, 0] = a
        matrix[..., 0, 3] = b
        matrix[..., 3, 0] = c
        matrix[..., 3, 3] = d
        matrix[..., 1, 1] = w
        matrix[..., 1, 2] = x
        matrix[..., 2, 1] = y
        matrix[..., 2, 2] = z
        if len(batch_sizes) == 0:
            matrix = matrix[0]
        return matrix
