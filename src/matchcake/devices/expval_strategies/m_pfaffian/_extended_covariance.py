from typing import Sequence

import numpy as np
import pennylane as qml
import torch

from ....typing import TensorLike
from ....utils.math import convert_and_cast_like


def displacement_vector(
    product_state_amplitudes: TensorLike,
    wires: Sequence[int],
) -> torch.Tensor:
    """Return d in R^{2n} with d[mu] = <c_mu>_rho for rho = product state.

    The product state is |psi_prod> = prod_k (a_k |0> + b_k |1>),
    where product_state_amplitudes[k, :] = (a_k, b_k).

    JW convention:
        <c_{2k}>   = (prod_{l<k} <Z_l>) * <X_k>
        <c_{2k+1}> = (prod_{l<k} <Z_l>) * <Y_k>

    :param product_state_amplitudes: Complex tensor of shape (n, 2).
    :param wires: Wire labels (unused for the computation; here for API consistency).
    :return: Real tensor of shape (2n,).
    """
    psi = torch.as_tensor(
        np.array(product_state_amplitudes)
        if not isinstance(product_state_amplitudes, torch.Tensor)
        else product_state_amplitudes,
        dtype=torch.complex128,
    )
    alpha = psi[..., 0]  # (..., n)
    beta = psi[..., 1]  # (..., n)

    x = 2.0 * torch.real(torch.conj(alpha) * beta)
    y = 2.0 * torch.imag(torch.conj(alpha) * beta)
    z = torch.abs(alpha) ** 2 - torch.abs(beta) ** 2  # (..., n), real

    n = psi.shape[-2]
    batch_shape = psi.shape[:-2]

    # Exclusive prefix product: z_prod[k] = z[0] * ... * z[k-1], z_prod[0] = 1.
    if n > 1:
        z_prod = torch.cat(
            [
                torch.ones(batch_shape + (1,), dtype=torch.float64, device=psi.device),
                torch.cumprod(z[..., :-1], dim=-1),
            ],
            dim=-1,
        )  # (..., n)
    else:
        z_prod = torch.ones(batch_shape + (n,), dtype=torch.float64, device=psi.device)

    d = torch.zeros(batch_shape + (2 * n,), dtype=torch.float64, device=psi.device)
    d[..., 0::2] = z_prod * x  # <c_{2k}>   = z_prod[k] * <X_k>
    d[..., 1::2] = z_prod * y  # <c_{2k+1}> = z_prod[k] * <Y_k>

    return d


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
    cov_matrix_t = torch.as_tensor(qml.math.real(cov_matrix), dtype=torch.float64)
    d = torch.as_tensor(
        np.array(displacement) if not isinstance(displacement, torch.Tensor) else displacement,
        dtype=torch.float64,
    )
    batch = cov_matrix_t.shape[:-2]

    d_col = d.unsqueeze(-1)  # (..., 2n, 1)
    neg_d_row = -d.unsqueeze(-2)  # (..., 1, 2n)
    zero_corner = torch.zeros(*batch, 1, 1, dtype=torch.float64, device=cov_matrix_t.device)

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
    Q_t = torch.as_tensor(qml.math.real(Q), dtype=torch.float64)
    batch = Q_t.shape[:-2]
    two_n = Q_t.shape[-1]

    zeros_col = torch.zeros(*batch, two_n, 1, dtype=torch.float64, device=Q_t.device)
    zeros_row = torch.zeros(*batch, 1, two_n, dtype=torch.float64, device=Q_t.device)
    one_corner = torch.ones(*batch, 1, 1, dtype=torch.float64, device=Q_t.device)

    top = torch.cat([Q_t, zeros_col], dim=-1)  # (..., 2n, 2n+1)
    bottom = torch.cat([zeros_row, one_corner], dim=-1)  # (..., 1,  2n+1)
    tilde_Q = torch.cat([top, bottom], dim=-2)  # (..., 2n+1, 2n+1)

    return convert_and_cast_like(tilde_Q, Q)
