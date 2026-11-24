import numpy as np
import pennylane as qml
import torch

from ....typing import TensorLike
from ....utils.math import convert_and_cast_like
from ....utils.torch_utils import infer_real_dtype


def extended_covariance_matrix(
    cov_matrix: TensorLike,
    displacement: TensorLike,
) -> TensorLike:
    """Return the extended covariance matrix in R^{(2n+1) x (2n+1)} with the parity index at 2n.

    Layout:
        ext_cov_matrix = [[ cov_matrix,  d   ],
                          [ -d^T,        0   ]]

    :param cov_matrix: (..., 2n, 2n) real antisymmetric covariance matrix.
    :param displacement: (..., 2n) real displacement vector d[mu] = <c_mu>.
    :return: (..., 2n+1, 2n+1) extended covariance matrix.
    """
    r_dtype = infer_real_dtype(cov_matrix)
    cov_matrix_t = torch.as_tensor(qml.math.real(cov_matrix), dtype=r_dtype)
    d = torch.as_tensor(
        np.array(displacement) if not isinstance(displacement, torch.Tensor) else displacement,
        dtype=r_dtype,
    )
    batch = cov_matrix_t.shape[:-2]

    d_col = d.unsqueeze(-1)  # (..., 2n, 1)
    neg_d_row = -d.unsqueeze(-2)  # (..., 1, 2n)
    zero_corner = torch.zeros(*batch, 1, 1, dtype=r_dtype, device=cov_matrix_t.device)

    top = torch.cat([cov_matrix_t, d_col], dim=-1)  # (..., 2n, 2n+1)
    bottom = torch.cat([neg_d_row, zero_corner], dim=-1)  # (..., 1,  2n+1)
    tilde = torch.cat([top, bottom], dim=-2)  # (..., 2n+1, 2n+1)

    return convert_and_cast_like(tilde, cov_matrix)


def sptm_lift(Q: TensorLike) -> TensorLike:
    """Lift a matchgate SPTM Q in O(2n) to tilde_Q in O(2n+1).

    Matchgate evolution preserves total parity, so the parity index at position 2n
    transforms trivially:

        tilde_Q = [[ Q, 0 ],
                   [ 0, 1 ]]

    :param Q: (..., 2n, 2n) orthogonal SPTM.
    :return: (..., 2n+1, 2n+1) lifted SPTM.
    """
    r_dtype = infer_real_dtype(Q)
    Q_t = torch.as_tensor(qml.math.real(Q), dtype=r_dtype)
    batch = Q_t.shape[:-2]
    two_n = Q_t.shape[-1]

    zeros_col = torch.zeros(*batch, two_n, 1, dtype=r_dtype, device=Q_t.device)
    zeros_row = torch.zeros(*batch, 1, two_n, dtype=r_dtype, device=Q_t.device)
    one_corner = torch.ones(*batch, 1, 1, dtype=r_dtype, device=Q_t.device)

    top = torch.cat([Q_t, zeros_col], dim=-1)  # (..., 2n, 2n+1)
    bottom = torch.cat([zeros_row, one_corner], dim=-1)  # (..., 1,  2n+1)
    tilde_Q = torch.cat([top, bottom], dim=-2)  # (..., 2n+1, 2n+1)

    return convert_and_cast_like(tilde_Q, Q)
