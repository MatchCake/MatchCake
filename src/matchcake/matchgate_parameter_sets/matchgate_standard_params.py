from typing import Optional

import numpy as np
import pennylane as qml
import torch
from dataclasses import dataclass
from pennylane.typing import TensorLike

from matchcake.utils.torch_utils import to_tensor


@dataclass
class MatchgateParams:
    ...

@dataclass
class MatchgateStandardParams(MatchgateParams):
    r"""
    Matchgate standard parameters.

    They are the parameters of a Matchgate operation in the standard form which is a 4x4 matrix

    .. math::

            \begin{bmatrix}
                a & 0 & 0 & b \\
                0 & w & x & 0 \\
                0 & y & z & 0 \\
                c & 0 & 0 & d
            \end{bmatrix}

        where :math:`a, b, c, d, w, x, y, z` are the parameters.
    """
    a: TensorLike
    b: TensorLike
    c: TensorLike
    d: TensorLike
    w: TensorLike
    x: TensorLike
    y: TensorLike
    z: TensorLike

    @classmethod
    def from_sub_matrices(
            cls,
            outer_matrix: TensorLike,
            inner_matrix: TensorLike,
    ):
        return cls(
            a=outer_matrix[..., 0, 0],
            b=outer_matrix[..., 0, 1],
            c=outer_matrix[..., 1, 0],
            d=outer_matrix[..., 1, 1],
            w=inner_matrix[..., 0, 0],
            x=inner_matrix[..., 0, 1],
            y=inner_matrix[..., 1, 0],
            z=inner_matrix[..., 1, 1],
        )

    def __post_init__(self):
        self._check_is_matchgate()

    def matrix(
            self,
            dtype: torch.dtype = torch.complex128,
            device: Optional[torch.device] = None,
    ):
        shapes = [
            qml.math.shape(p)
            for p in [self.a, self.b, self.c, self.d, self.w, self.x, self.y, self.z]
            if p is not None
        ]
        batch_sizes = list(set([s[0] for s in shapes if len(s) > 0]))
        assert len(batch_sizes) <= 1, f"Expect the same batch size for every parameters. Got: {batch_sizes}."
        batch_size = batch_sizes[0] if len(batch_sizes) > 0 else 1
        a, b, c, d, w, x, y, z = [
            (
                to_tensor(p, dtype=dtype, device=device)
                if p is not None
                else torch.zeros((batch_size,), dtype=dtype, device=device)
            )
            for p in [self.a, self.b, self.c, self.d, self.w, self.x, self.y, self.z]
        ]
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

    def adjoint(self) -> "MatchgateStandardParams":
        r"""
        Return the adjoint version of the parameters.

        :return: The adjoint parameters.
        """
        return MatchgateStandardParams(
            a=qml.math.conjugate(self.a),
            b=qml.math.conjugate(self.c),
            c=qml.math.conjugate(self.b),
            d=qml.math.conjugate(self.d),
            w=qml.math.conjugate(self.w),
            x=qml.math.conjugate(self.y),
            y=qml.math.conjugate(self.x),
            z=qml.math.conjugate(self.z),
        )

    def _check_m_m_dagger_constraint(self) -> bool:
        with torch.no_grad():
            m_m_dagger = torch.einsum("...ij,...kj->...ik", self.matrix(), torch.conj(self.matrix()))
            expected_zero = m_m_dagger - torch.eye(4)
            return torch.allclose(expected_zero, torch.zeros_like(expected_zero), atol=1e-5)

    def _check_m_dagger_m_constraint(self) -> bool:
        with torch.no_grad():
            m_dagger_m = torch.einsum("...ji,...jk->...ik", torch.conj(self.matrix()), self.matrix())
            expected_zero = m_dagger_m - torch.eye(4)
            return torch.allclose(expected_zero, torch.zeros_like(expected_zero), atol=1e-5)

    def _check_det_constraint(self) -> bool:
        with torch.no_grad():
            outer_det = self.a * self.d - self.c * self.b
            inner_det = self.w * self.z - self.y * self.x
            return qml.math.allclose(outer_det, inner_det, atol=1e-5)

    def _check_is_matchgate(self):
        if not self._check_m_m_dagger_constraint():
            raise ValueError(r"The matchgate does not satisfy the M M^\dagger constraint.")
        if not self._check_m_dagger_m_constraint():
            raise ValueError(r"The matchgate does not satisfy the M^\dagger M constraint.")  # pragma: no cover
        if not self._check_det_constraint():
            raise ValueError(r"The matchgate does not satisfy the determinant constraint.")
