import numpy as np
import pennylane as qml
import torch

from ...typing import TensorLike
from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import infer_real_dtype

# |d| below this snaps to the zero-displacement (basis-state) path: b0 = cov @ d / m divides by
# m -> 0 as |d| -> 0, so the ancilla block is ill-conditioned below ~1e-6 (verified down to this
# norm in verify_edge_cases.py).
DISPLACEMENT_TOL = 1e-6


def lift_sptm(sptm: TensorLike) -> TensorLike:
    r"""Lift a ``(..., 2n, 2n)`` matchgate SPTM to the even ``(2n+2)`` frame ``sptm (+) I_2``.

    The two ancilla modes (the ``b0`` mode at index ``2n`` and the parity marker at ``2n+1``) are
    inert under matchgate evolution, so they ride along as an identity block:

    .. math::
        \widetilde{Q} = \begin{bmatrix} Q & 0 \\ 0 & I_2 \end{bmatrix}.

    This is the ``(2n+2)`` analogue of
    :func:`...expval_strategies.m_pfaffian._extended_covariance.sptm_lift` (which appends a single
    parity row/col for the odd ``(2n+1)`` encoding). Supports arbitrary leading batch dimensions.

    :param sptm: Orthogonal single-particle transition matrix of shape ``(..., 2n, 2n)``.
    :return: Lifted SPTM of shape ``(..., 2n+2, 2n+2)``, on the backend/dtype of ``sptm``.
    :rtype: TensorLike
    """
    real_dtype = infer_real_dtype(sptm)
    sptm_real = torch.as_tensor(qml.math.real(sptm), dtype=real_dtype)
    batch = sptm_real.shape[:-2]
    two_n = sptm_real.shape[-1]

    lifted = torch.zeros(*batch, two_n + 2, two_n + 2, dtype=real_dtype, device=sptm_real.device)
    lifted[..., :two_n, :two_n] = sptm_real
    lifted[..., two_n, two_n] = 1.0
    lifted[..., two_n + 1, two_n + 1] = 1.0
    return convert_and_cast_like(lifted, sptm)


def lift_from_product_state(covariance: TensorLike, displacement: TensorLike) -> TensorLike:
    r"""Even ``(2n+2)`` orthogonal lift of a product state from its covariance and displacement.

    Given the physical covariance and displacement of a product state, returns the real orthogonal
    lift ``lifted`` of shape ``(2n+2, 2n+2)`` (``lifted @ lifted = -I``) whose physical block is the
    covariance, with the ``b0`` ancilla mode at index ``2n`` and the parity marker (``= -d``) at
    index ``2n+1`` (swap_injection_theory.md eq 22):

    .. math::
        M = \begin{bmatrix} \Lambda & b_0 & -d \\ -b_0^T & 0 & -m \\ d^T & m & 0 \end{bmatrix},
        \quad m^2 = -\frac{d^T \Lambda^2 d}{d^T d}, \quad b_0 = \frac{\Lambda d}{m}.

    When ``|displacement| < DISPLACEMENT_TOL`` the state is a basis state (up to per-qubit phase) and
    the ancilla decouples: ``lifted = covariance (+) [[0, 1], [-1, 0]]``. This is the zero-
    displacement special case, so a single code path serves basis and product inputs.

    :param covariance: Real antisymmetric covariance matrix of shape ``(2n, 2n)``.
    :param displacement: Real displacement vector ``d[mu] = <c_mu>`` of shape ``(2n,)``.
    :return: Real orthogonal lift of shape ``(2n+2, 2n+2)``, on the backend/dtype of ``covariance``.
    :rtype: TensorLike
    :raises AssertionError: if ``rank(I + covariance^2) > 2`` (the input is not a product state, so a
        single ancilla mode is insufficient).
    """
    real_dtype = infer_real_dtype(covariance)
    covariance_real = torch.as_tensor(qml.math.real(covariance), dtype=real_dtype)
    displacement_real = torch.as_tensor(qml.math.real(displacement), dtype=real_dtype)
    if covariance_real.ndim != 2:
        raise NotImplementedError(
            "lift_from_product_state expects an unbatched (2n, 2n) covariance; the initial product "
            "state is lifted once and then propagated, so batched parameters enter via the SPTMs."
        )
    dim = covariance_real.shape[-1]
    lifted = torch.zeros(dim + 2, dim + 2, dtype=real_dtype, device=covariance_real.device)
    lifted[:dim, :dim] = covariance_real

    norm_sq = torch.dot(displacement_real, displacement_real)
    if float(norm_sq) < DISPLACEMENT_TOL**2:
        lifted[dim, dim + 1], lifted[dim + 1, dim] = 1.0, -1.0
        return convert_and_cast_like(lifted, covariance)

    # Product-state invariant: a single ancilla mode always suffices (verified for n up to 6).
    rank = int(np.linalg.matrix_rank(qml.math.toarray(covariance_real @ covariance_real) + np.eye(dim), tol=1e-9))
    assert rank <= 2, (
        f"lift_from_product_state: rank(I + covariance^2) = {rank} > 2; the input is not a product "
        f"state, so a single ancilla mode is insufficient."
    )

    scale_sq = -(displacement_real @ (covariance_real @ (covariance_real @ displacement_real))) / norm_sq
    scale = torch.sqrt(torch.clamp(scale_sq, min=0.0))
    ancilla_coupling = (covariance_real @ displacement_real) / scale
    lifted[:dim, dim] = ancilla_coupling
    lifted[dim, :dim] = -ancilla_coupling
    lifted[:dim, dim + 1] = -displacement_real
    lifted[dim + 1, :dim] = displacement_real
    lifted[dim, dim + 1] = scale
    lifted[dim + 1, dim] = -scale
    return convert_and_cast_like(lifted, covariance)
