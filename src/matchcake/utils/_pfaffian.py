import threading
from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch_pfaffian
from pennylane.typing import TensorLike

from . import torch_utils
from .math import convert_and_cast_like
from .torch_utils import infer_complex_dtype, infer_real_dtype

_pfaffian_epsilon_lock = threading.Lock()


def signed_pfaffian(matrix: TensorLike, dtype: Optional[torch.dtype] = None) -> TensorLike:
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
    return pfaffian(matrix, sign=True, dtype=dtype)


def signed_pfaffian_complex(matrix: TensorLike, dtype: Optional[torch.dtype] = None) -> TensorLike:
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
    result = torch_pfaffian.pfaffian(matrix_t, sign=True)
    return convert_and_cast_like(result, matrix_t)


def sector_pfaffian_features(
    cov_matrix: TensorLike,
    index_sets: np.ndarray,
    dtype: Optional[torch.dtype] = None,
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
        result = pfaffian(submatrices, sign=True)  # (..., n_terms)
    return convert_and_cast_like(result, cov_matrix)


def pfaffian(
    matrix: TensorLike,
    sign: bool = False,
    epsilon: float = 1e-32,
    dtype: Optional[torch.dtype] = None,
) -> TensorLike:
    """
    Compute the Pfaffian of a real or complex skew-symmetric matrix ``A`` (``A = -A^T``).

    Delegates to TorchPfaffian. When ``sign`` is ``False`` (default) the magnitude
    ``sqrt(|det(A)|)`` is returned, which is the quantity needed for probability computations
    where the sign is irrelevant. When ``sign`` is ``True`` the signed Pfaffian is returned.

    Note: with a complex ``matrix`` this function may not behave as expected. The signed path
    (``sign=True``) relies on a kernel that discards the imaginary part without warning, so it
    returns the Pfaffian of the real part only. In normal use the signed path receives real
    matrices, and the magnitude path (``sign=False``) handles complex inputs correctly. If the
    complex behaviour is a problem for your use case, please open an issue.

    :param matrix: Matrix to compute the Pfaffian of, of shape ``(..., 2n, 2n)``.
    :param sign: When ``True`` return the signed Pfaffian, otherwise its magnitude. Defaults
        to ``False``.
    :param epsilon: Floor applied to the determinant magnitude in the ``sign=False`` path to
        avoid numerical instabilities. Defaults to ``1e-32``.
    :param dtype: Optional working dtype to cast ``matrix`` to before the computation.
        Defaults to ``None`` (the input dtype is preserved).
    :return: Pfaffian of the matrix, of shape ``(...,)``.
    :rtype: TensorLike
    """
    matrix_t = torch_utils.to_tensor(matrix, dtype=dtype)
    if sign:
        result = torch_pfaffian.pfaffian(matrix_t, sign=True)
    else:
        # ``PfaffianDet`` differentiates straight through ``det`` and ``sqrt``; its gradient
        # stays consistent with finite differences for complex matrices, whereas the analytic
        # backward of the gradient-enabled magnitude strategy does not. The global ``EPSILON``
        # it reads is mutated under a lock for thread safety.
        # ``PfaffianDet`` is also faster in general in CPU and can be used with CUDA.
        with _pfaffian_epsilon_lock:
            torch_pfaffian.PfaffianStrategy.EPSILON = epsilon
            result = torch_pfaffian.PfaffianDet.apply(matrix_t)
    return convert_and_cast_like(result, matrix)
