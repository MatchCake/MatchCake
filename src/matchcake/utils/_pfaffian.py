from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch_pfaffian
from pennylane.typing import TensorLike

from . import torch_utils
from .math import convert_and_cast_like
from .torch_utils import infer_complex_dtype, infer_real_dtype


def signed_pfaffian(matrix: TensorLike, dtype: Optional[torch.dtype] = None, **kwargs) -> TensorLike:
    """
    Compute the signed Pfaffian of a real antisymmetric matrix (or batch).

    Thin wrapper over :func:`pfaffian` with ``sign=True`` that infers a real working
    precision from ``matrix``. The signed Pfaffian uses the native Rust Parlett-Reid kernel
    on CPU and a device-native PyTorch fallback otherwise, supports arbitrary leading batch
    dimensions, and is autograd-safe on the skew-symmetric manifold.

    :param matrix: Real antisymmetric matrix of even size ``(..., 2k, 2k)``.
    :param dtype: Real working precision for the internal computation. Defaults to
        ``None``, in which case the precision is inferred from ``matrix`` (e.g. a
        ``float32``/``complex64`` input keeps ``float32`` internals). Pass an explicit
        real dtype to override.
    :return: Signed Pfaffian of shape ``(...,)``.
    :rtype: TensorLike
    """
    if dtype is None:
        dtype = infer_real_dtype(matrix)
    return pfaffian(matrix, sign=True, dtype=dtype, **kwargs)


def signed_pfaffian_complex(matrix: TensorLike, dtype: Optional[torch.dtype] = None, **kwargs) -> TensorLike:
    """
    Compute the **signed** Pfaffian of a **complex** antisymmetric matrix (or batch).

    Returns the signed Pfaffian ``Pf`` (not its magnitude ``|Pf|``) of a complex skew-symmetric
    matrix of shape ``(..., 2k, 2k)``, preserving the imaginary part. This is the quantity the
    SWAP-injection branch formalism needs: its transition Pfaffians are genuinely complex and the
    branch interference carries the sign, so neither the magnitude path nor the real signed path
    is usable.

    The computation delegates to ``torch_pfaffian.pfaffian(matrix, sign=True)``, whose dispatch
    routes complex inputs to the device-native Parlett-Reid strategy (batched, signed, imaginary
    part preserved). MatchCake's other Pfaffian backends (``det``, ``cuda_det``, ``PfaffianFDBPf``)
    return the **unsigned** Pfaffian and are unusable here (the sign is required for ``m >= 4``).

    :param matrix: Complex antisymmetric matrix of even size ``(..., 2k, 2k)``.
    :param dtype: Complex working precision for the internal computation. Defaults to ``None``, in
        which case the precision is inferred from ``matrix`` (e.g. a ``complex64`` input keeps
        ``complex64`` internals). Pass an explicit complex dtype to override.
    :return: Complex signed Pfaffian of shape ``(...,)``.
    :rtype: TensorLike
    """
    if dtype is None:
        dtype = infer_complex_dtype(matrix)
    matrix_t = torch_utils.to_tensor(matrix, dtype=dtype)
    result = pfaffian(matrix_t, sign=True, dtype=dtype, **kwargs)
    return convert_and_cast_like(result, matrix_t)


def sector_pfaffian_features(
    cov_matrix: TensorLike, index_sets: np.ndarray, dtype: Optional[torch.dtype] = None, **kwargs
) -> TensorLike:
    """
    Return a ``(..., n_terms)`` tensor of signed Pfaffians of principal submatrices.

    Each submatrix is the ``(submatrix_size, submatrix_size)`` principal minor of
    ``cov_matrix`` indexed by a row of ``index_sets``. The signed Pfaffians are computed
    in a single vectorized :func:`pfaffian` call with ``sign=True``, which carries the
    gradient on the skew-symmetric manifold.

    :param cov_matrix: ``(..., D, D)`` antisymmetric matrix (or batch).
    :param index_sets: ``(n_terms, submatrix_size)`` integer array of Majorana index tuples.
    :param dtype: Real working precision for the internal computation. Defaults to
        ``None``, in which case the precision is inferred from ``cov_matrix`` (e.g. a
        ``float32``/``complex64`` input keeps ``float32`` internals, avoiding the
        ~2x memory of forcing ``float64``). Pass an explicit real dtype to override.
    :return: ``(..., n_terms)`` Pfaffians.
    :rtype: TensorLike
    """
    if dtype is None:
        dtype = infer_real_dtype(cov_matrix)
    cov_matrix_t = torch.as_tensor(qml.math.real(cov_matrix), dtype=dtype)
    index_sets = np.asarray(index_sets)  # (n_terms, submatrix_size)
    n_terms, submatrix_size = index_sets.shape
    row_indices = index_sets[:, :, None]  # (n_terms, submatrix_size, 1)
    col_indices = index_sets[:, None, :]  # (n_terms, 1, submatrix_size)
    submatrices = cov_matrix_t[..., row_indices, col_indices]  # (..., n_terms, submatrix_size, submatrix_size)

    if submatrix_size == 2:
        # Fast path: Pf of 2x2 [[0, a], [-a, 0]] = a.
        result = submatrices[..., 0, 1]  # (..., n_terms)
    else:
        result = pfaffian(submatrices, sign=True, **kwargs)  # (..., n_terms)
    return convert_and_cast_like(result, cov_matrix)


def _pfaffian_kernel(matrix_t: torch.Tensor, sign: bool, epsilon: float) -> torch.Tensor:
    """Compute the Pfaffian of a batch of matrices already cast to a torch tensor.

    :param matrix_t: Skew-symmetric matrix (or batch) of shape ``(..., 2n, 2n)``.
    :type matrix_t: torch.Tensor
    :param sign: When ``True`` return the signed Pfaffian, otherwise its magnitude.
    :type sign: bool
    :param epsilon: Floor applied to the Pfaffian magnitude in the ``sign=False`` path; the magnitude
        ``|Pf|`` is floored at ``sqrt(epsilon)``.
    :type epsilon: float
    :return: Pfaffian of shape ``(...,)``.
    :rtype: torch.Tensor
    """
    if sign:
        return torch_pfaffian.pfaffian(matrix_t, sign=True)
    # Magnitude path: |Pf| = sqrt(|det|) floored at sqrt(epsilon), with epsilon a local argument.
    # Kept det-based through the new dispatch because the log-domain magnitude kernel is slower here with no
    # gradient benefit (the value and gradient are identical to this floor).
    return torch.sqrt(torch.clamp(torch.abs(torch.linalg.det(matrix_t)), min=epsilon))


def pfaffian(
    matrix: TensorLike,
    sign: bool = False,
    epsilon: float = 1e-32,
    dtype: Optional[torch.dtype] = None,
    chunk_size: Optional[int] = None,
) -> TensorLike:
    """
    Compute the Pfaffian of a real or complex skew-symmetric matrix ``A`` (``A = -A^T``).

    Delegates to TorchPfaffian. When ``sign`` is ``False`` (default) the magnitude ``|Pf(A)|`` is
    returned, floored at ``sqrt(epsilon)``, which is the quantity needed for probability computations
    where the sign is
    irrelevant. When ``sign`` is ``True`` the signed Pfaffian is returned. Both paths preserve the
    imaginary part of a complex ``matrix``: the TorchPfaffian Parlett-Reid strategies compute the
    complex signed Pfaffian without discarding it.

    For a large batch the pfaffian workspace and its backward graph dominate
    memory and can exceed device memory. Pass ``chunk_size`` to bound that footprint: the
    leading batch axis is then processed in slices of at most ``chunk_size`` matrices, each
    reduced independently and concatenated, instead of all at once.

    :param matrix: Matrix to compute the Pfaffian of, of shape ``(..., 2n, 2n)``.
    :param sign: When ``True`` return the signed Pfaffian, otherwise its magnitude. Defaults
        to ``False``.
    :param epsilon: Floor applied to the Pfaffian magnitude in the ``sign=False`` path (``|Pf|`` is
        floored at ``sqrt(epsilon)``). Passed per call, so concurrent calls with different values do
        not interfere. Defaults to ``1e-32``.
    :param dtype: Optional working dtype to cast ``matrix`` to before the computation.
        Defaults to ``None`` (the input dtype is preserved).
    :param chunk_size: Maximum number of matrices to reduce at once along the flattened leading
        batch axis. Defaults to ``None`` (the whole batch is reduced in one shot).
    :return: Pfaffian of the matrix, of shape ``(...,)``.
    :rtype: TensorLike
    """
    matrix_t = torch_utils.to_tensor(matrix, dtype=dtype)
    batch_shape = matrix_t.shape[:-2]

    if chunk_size is not None and len(batch_shape) > 0 and batch_shape.numel() > chunk_size:
        flat = matrix_t.reshape(-1, matrix_t.shape[-2], matrix_t.shape[-1])  # (B, 2n, 2n)
        pieces = [
            _pfaffian_kernel(flat[start : start + chunk_size], sign, epsilon)
            for start in range(0, flat.shape[0], chunk_size)
        ]
        result = torch.cat(pieces, dim=0).reshape(batch_shape)
    else:
        result = _pfaffian_kernel(matrix_t, sign, epsilon)
    return convert_and_cast_like(result, matrix)
