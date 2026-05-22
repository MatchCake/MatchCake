import threading
from typing import Literal, Optional

import numpy as np
import pennylane as qml
import torch
import torch_pfaffian
import tqdm
from pennylane.typing import TensorLike

from . import torch_utils
from .math import convert_and_cast_like


def signed_pfaffian(matrix: TensorLike) -> TensorLike:
    """
    Compute the signed Pfaffian of a real antisymmetric matrix (or batch).

    Processes each element of the batch independently using the Parlett-Reid
    algorithm (skew-tridiagonalization with partial pivoting). Supports
    arbitrary leading batch dimensions.

    :param matrix: Real antisymmetric matrix of even size (..., 2k, 2k).
    :return: Signed Pfaffian of shape (...,).
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.as_tensor(qml.math.real(matrix), dtype=torch.float64)
    A = matrix.to(torch.float64)
    shape = A.shape
    n = shape[-1]

    if n == 0:
        return torch.ones(shape[:-2], dtype=A.dtype, device=A.device)
    if n % 2 == 1:
        return torch.zeros(shape[:-2], dtype=A.dtype, device=A.device)

    batch_shape = shape[:-2]
    # Flatten batch dims for uniform processing
    A_flat = A.reshape(-1, n, n).clone()  # (B, n, n)
    B = A_flat.shape[0]
    pf_flat = torch.ones(B, dtype=A.dtype, device=A.device)

    for b in range(B):
        M = A_flat[b]
        sign = 1.0
        for i in range(0, n - 2, 2):
            # Pivot: find row with max |M[j, i]| for j > i+1
            col_abs = M[i + 2:, i].abs()
            p_rel = int(torch.argmax(col_abs).item())
            p = p_rel + i + 2
            if p != i + 1 and col_abs[p_rel].item() > abs(M[i + 1, i].item()):
                # Congruence swap: rows/cols i+1 <-> p
                M[[i + 1, p], :] = M[[p, i + 1], :]
                M[:, [i + 1, p]] = M[:, [p, i + 1]]
                sign *= -1.0
            # Schur complement elimination
            pivot_val = M[i + 1, i].item()
            if abs(pivot_val) < 1e-300:
                sign = 0.0
                break
            tau = M[i + 2:, i] / pivot_val          # (n-i-2,)
            row_ip1 = M[i + 1, i + 2:]               # (n-i-2,)
            col_ip1 = M[i + 2:, i + 1]               # (n-i-2,)
            M[i + 2:, i + 2:] += (
                torch.outer(tau, col_ip1) - torch.outer(col_ip1, tau)
            )
            M[i + 2:, i] = 0.0
            M[i, i + 2:] = 0.0
            M[i + 2:, i + 1] = 0.0
            M[i + 1, i + 2:] = 0.0

        # Product of super-diagonal entries
        pf_T = 1.0
        for i in range(0, n, 2):
            pf_T *= M[i, i + 1].item()
        pf_flat[b] = sign * pf_T

    result = pf_flat.reshape(batch_shape if batch_shape else ())
    return convert_and_cast_like(result, matrix)


def sector_pfaffian_features(
    cov_matrix: TensorLike,
    index_sets: np.ndarray,
) -> TensorLike:
    """
    Return (..., n_terms) tensor of signed Pfaffians of (submatrix_size x submatrix_size)
    principal submatrices.

    :param cov_matrix: (..., D, D) antisymmetric matrix (or batch).
    :param index_sets: (n_terms, submatrix_size) integer array of Majorana index tuples.
    :return: (..., n_terms) Pfaffians.
    """
    cov_matrix_t = torch.as_tensor(qml.math.real(cov_matrix), dtype=torch.float64)
    index_sets = np.asarray(index_sets)                      # (n_terms, submatrix_size)
    n_terms, submatrix_size = index_sets.shape
    row_indices = index_sets[:, :, None]                     # (n_terms, submatrix_size, 1)
    col_indices = index_sets[:, None, :]                     # (n_terms, 1, submatrix_size)
    submatrices = cov_matrix_t[..., row_indices, col_indices]  # (..., n_terms, submatrix_size, submatrix_size)

    if submatrix_size == 2:
        # Fast path: Pf of 2x2 [[0, a], [-a, 0]] = a
        result = submatrices[..., 0, 1]  # (..., n_terms)
    else:
        result = torch.stack(
            [signed_pfaffian(submatrices[..., k, :, :]) for k in range(n_terms)], dim=-1
        )
    return convert_and_cast_like(result, cov_matrix)

_pfaffian_fdbpf_lock = threading.Lock()


def pfaffian_by_det(
    __matrix: TensorLike,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
    epsilon: float = 1e-32,
) -> TensorLike:
    """
    Compute the Pfaffian of a skew-symmetric matrix using determinant-based methods.

    This function calculates the Pfaffian value of a skew-symmetric matrix (or a batch
    of such matrices) using the determinant of the matrix. If the matrix is of odd
    dimensions, the result is zero because skew-symmetric matrices of odd size
    always have a zero Pfaffian. The computation is optimized for different backends
    like "autograd", "numpy", and others.

    :param __matrix: Input skew-symmetric matrix or batch of skew-symmetric matrices.
    :param p_bar: Optional progress bar instance for updating progress.
    :param show_progress: Whether to display progress during computation. Defaults
        to False.
    :param epsilon: Minimum value to clip the determinant during the computation of
        the Pfaffian to avoid numerical instabilities. Defaults to 1e-32.
    :return: The computed Pfaffian value(s) of the input skew-symmetric matrix or batch.
    :rtype: TensorLike
    """
    shape = qml.math.shape(__matrix)
    p_bar = p_bar or tqdm.tqdm(total=1, disable=not show_progress)
    p_bar.set_description(f"[det] Computing determinant of {shape} matrix")

    # Quick return if possible
    if shape[-2] % 2 == 1:
        p_bar.set_description("Odd-sized matrix")
        p_bar.close()
        matrix = qml.math.cast(__matrix, dtype=complex)
        zero_like = convert_and_cast_like(0, matrix)
        pfaffian_val = qml.math.convert_like(np.ones(shape[:-2], dtype=complex), matrix)
        return pfaffian_val * zero_like * matrix[..., 0, 0]

    backend = qml.math.get_interface(__matrix)
    if backend in ["autograd", "numpy"]:
        det = qml.math.linalg.det(__matrix)
        pf = qml.math.sqrt(qml.math.clip(qml.math.abs(det), min=epsilon))
    else:
        det = qml.math.det(__matrix)
        pf = qml.math.sqrt(qml.math.clamp(qml.math.abs(det), min=epsilon))
    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian_by_det_cuda(
    __matrix: TensorLike,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
    epsilon: float = 1e-32,
) -> TensorLike:  # pragma: no cover
    shape = qml.math.shape(__matrix)
    p_bar = p_bar or tqdm.tqdm(total=1, disable=not show_progress)
    p_bar.set_description(f"[cuda_det] Computing determinant of {shape} matrix")
    backend = qml.math.get_interface(__matrix)
    cuda_det = torch.det(torch_utils.to_cuda(__matrix))
    if backend in ["torch"]:
        if __matrix.device.type == "cuda":
            det = cuda_det
        else:
            det = torch_utils.to_cpu(cuda_det)
    else:
        det = torch_utils.to_cpu(cuda_det)
    det = convert_and_cast_like(det, __matrix)

    det_backend = qml.math.get_interface(det)
    if det_backend in ["autograd", "numpy"]:
        pf = qml.math.sqrt(qml.math.clip(qml.math.abs(det), min=epsilon))
    else:
        pf = qml.math.sqrt(qml.math.clamp(qml.math.abs(det), min=epsilon))

    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian(
    __matrix: TensorLike,
    method: Literal["det", "cuda_det", "PfaffianFDBPf"] = "det",
    epsilon: float = 1e-32,
    p_bar: Optional[tqdm.tqdm] = None,
    show_progress: bool = False,
) -> TensorLike:
    """
    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T).

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: TensorLike
    :param method: Method to use to compute the pfaffian.
    :type method: Literal["det", "cuda_det", "PfaffianFDBPf"]
    :param epsilon: Tolerance for the determinant method
    :type epsilon: float
    :param p_bar: Progress bar
    :type p_bar: Optional[tqdm.tqdm]
    :param show_progress: Whether to show progress bar. If no progress bar is provided, a new one is created
    :type show_progress: bool
    :return: Pfaffian of the matrix
    :rtype: TensorLike
    """
    shape = qml.math.shape(__matrix)
    assert shape[-2] == shape[-1] > 0, "Matrix must be square"

    if method == "det":
        return pfaffian_by_det(__matrix, p_bar=p_bar, show_progress=show_progress, epsilon=epsilon)
    elif method == "cuda_det":  # pragma: no cover
        return pfaffian_by_det_cuda(__matrix, p_bar=p_bar, show_progress=show_progress, epsilon=epsilon)
    elif method == "PfaffianFDBPf":
        with _pfaffian_fdbpf_lock:
            torch_pfaffian.PfaffianStrategy.EPSILON = epsilon
            pf = torch_pfaffian.get_pfaffian_function(method)(torch_utils.to_tensor(__matrix, dtype=torch.complex128))
        return convert_and_cast_like(pf, __matrix)
    raise ValueError(f"Invalid method. Got {method}, must be 'det', 'cuda_det', or 'PfaffianFDBPf'.")
