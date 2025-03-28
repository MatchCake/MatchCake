from typing import Union, Literal, Optional
import pennylane as qml
import numpy as np
import warnings

import torch
import tqdm

from .math import convert_and_cast_like
from ..templates import TensorLike
from .torch_pfaffian import Pfaffian


def _pivot(__matrix, k, kp):
    matrix = qml.math.ones_like(__matrix) * __matrix
    matrix_shape = qml.math.shape(matrix)
    indexes_to_pivot = np.arange(matrix_shape[0])
    if len(matrix_shape) == 2:
        matrix[..., [kp, k + 1], k:] = matrix[..., [k + 1, kp], k:]
        matrix[..., k:, [kp, k + 1]] = matrix[..., k:, [k + 1, kp]]
        return matrix

    # interchange rows k+1 and kp
    kp_sub_matrix = matrix[indexes_to_pivot, kp, k:]
    temp = qml.math.ones_like(kp_sub_matrix) * kp_sub_matrix
    matrix[indexes_to_pivot, kp, k:] = matrix[indexes_to_pivot, k + 1, k:]
    matrix[indexes_to_pivot, k + 1, k:] = temp

    # Then interchange columns k+1 and kp
    kp_sub_matrix = matrix[indexes_to_pivot, k:, kp]
    temp = qml.math.ones_like(kp_sub_matrix) * kp_sub_matrix
    matrix[indexes_to_pivot, k:, kp] = matrix[indexes_to_pivot, k:, k + 1]
    matrix[indexes_to_pivot, k:, k + 1] = temp
    return matrix


def _compute_gauss_vector(__matrix, k):
    zero_like = convert_and_cast_like(0, __matrix)
    tau_norm = __matrix[..., k, k + 1][..., None]
    zero_mask = qml.math.isclose(tau_norm, zero_like)
    tau = qml.math.where(zero_mask, zero_like, __matrix[..., k, k + 2:] / tau_norm)
    return tau


def _update_matrix_block_kp2_kp2(__matrix, k, tau):
    add_matrix = qml.math.zeros_like(__matrix)
    add_matrix[..., k + 2:, k + 2:] += (
        qml.math.einsum("...i,...j->...ij", tau, __matrix[..., k + 2:, k + 1])
        -
        qml.math.einsum("...i,...j->...ij", __matrix[..., k + 2:, k + 1], tau)
    )
    return __matrix + add_matrix


def batch_pfaffian_ltl(
        __matrix: TensorLike,
        overwrite_input: bool = False,
        test_input: bool = False,
        p_bar: Optional[tqdm.tqdm] = None,
        show_progress: bool = False
) -> Union[float, complex, TensorLike]:
    r"""
    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.

    This code is adapted of the function `pfaffian_LTL`
    from https://github.com/basnijholt/pfapack/blob/master/pfapack/pfaffian.py.

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: TensorLike
    :param overwrite_input: Whether to overwrite the input matrix
    :type overwrite_input: bool
    :param test_input: Whether to test the input matrix for skew-symmetry
    :type test_input: bool
    :param p_bar: Progress bar
    :type p_bar: Optional[tqdm.tqdm]
    :param show_progress: Whether to show progress bar. If no progress bar is provided, a new one is created
    :type show_progress: bool

    :return: Pfaffian of the matrix
    :rtype: Union[float, complex, TensorLike]
    """
    if overwrite_input:
        matrix = __matrix
    else:
        matrix = qml.math.ones_like(__matrix) * __matrix
    shape = qml.math.shape(matrix)
    n, m = shape[-2:]
    p_bar = p_bar or tqdm.tqdm(range(0, n - 1, 2), disable=not show_progress)

    if test_input:
        p_bar.set_description("Testing input matrix")
        # Check if matrix is square
        assert shape[-2] == shape[-1] > 0
        # Check if it's skew-symmetric
        matrix_t = qml.math.einsum("...ij->...ji", matrix)
        assert qml.math.allclose(matrix, -matrix_t)

    matrix = qml.math.cast(matrix, dtype=complex)
    zero_like = convert_and_cast_like(0, matrix)
    pfaffian_val = qml.math.convert_like(np.ones(shape[:-2], dtype=complex), matrix)

    # Quick return if possible
    if n % 2 == 1:
        p_bar.n = n//2
        p_bar.set_description("Odd-sized matrix")
        p_bar.close()
        return pfaffian_val * zero_like * matrix[..., 0, 0]  # 0.0 but with require grad if needed

    p_bar.set_description(f"Computing Pfaffian of {shape} matrix")
    for k in p_bar:
        # First, find the largest entry in A[k+1:,k] and permute it to A[k+1,k]
        kp = k + 1 + qml.math.abs(matrix[..., k + 1:, k]).argmax(-1)
        kp1 = qml.math.convert_like(k + 1, kp)

        # Check if we need to pivot
        pivot_condition = ~qml.math.isclose(kp, kp1)
        # interchange rows and cols k+1 and kp (pivot if needed)
        matrix = qml.math.where(pivot_condition[..., None, None], _pivot(matrix, k, kp), matrix)
        # every interchange corresponds to a "-" in det(P)
        pfaffian_val *= qml.math.where(pivot_condition, -1.0, 1.0)

        # if we encounter a zero on the super/subdiagonal, the Pfaffian is 0
        zero_ss_condition = qml.math.isclose(matrix[..., k + 1, k], zero_like)
        pfaffian_val *= qml.math.where(zero_ss_condition, zero_like, matrix[..., k, k + 1])
        if qml.math.all(zero_ss_condition):
            p_bar.n = n//2
            p_bar.set_description("Pfaffian is zero")
            p_bar.close()
            return pfaffian_val

        if k + 2 < n:
            tau = _compute_gauss_vector(matrix, k)
            # Update the matrix block A(k+2:,k+2)
            matrix = _update_matrix_block_kp2_kp2(matrix, k, tau)

    p_bar.set_description("Pfaffian computed")
    p_bar.close()
    return pfaffian_val


def batch_householder_complex(x: TensorLike):
    """(v, tau, alpha) = householder_real(x)

    Compute a Householder transformation such that
    (1-tau v v^T) x = alpha e_1
    where x and v a complex vectors, tau is 0 or 2, and
    alpha a complex number (e_1 is the first unit vector)
    """
    sigma = qml.math.einsum("...i,...i->...", qml.math.conjugate(x[..., 1:]), x[..., 1:])

    # if sigma == 0:
    #     return qml.math.zeros_like(x), 0, x[..., 0]
    # else:
    norm_x = qml.math.sqrt(qml.math.conjugate(x[..., 0]) * x[..., 0] + sigma)

    v = qml.math.ones_like(x) * x
    phase = qml.math.exp(1j * qml.math.arctan2(qml.math.imag(x[..., 0]), qml.math.real(x[..., 0])))
    v[..., 0] = v[..., 0] + phase * norm_x

    v_frobenius_norm = qml.math.einsum("...i,...i->...", qml.math.conjugate(v), v) ** 0.5
    v = v / v_frobenius_norm[..., None]

    tau = convert_and_cast_like(2.0+0.0j, x)
    return v, tau, -phase * norm_x


def _eliminate_ith_column_householder(matrix, i, alpha):
    """
    Eliminate the i-th column of the matrix using the Householder transformation.
    Equivalent to:

        matrix[..., i + 1, i] = alpha
        matrix[..., i, i + 1] = -alpha
        matrix[..., i + 2:, i] = zero_like
        matrix[..., i, i + 2:] = zero_like

    :param matrix:
    :param i:
    :param alpha:
    :return:
    """
    zero_like = convert_and_cast_like(0, matrix)

    where_mask = qml.math.cast(qml.math.zeros_like(matrix), dtype=bool)
    where_mask[..., i + 1, i] = True
    where_mask[..., i, i + 1] = True
    where_mask[..., i + 2:, i] = True
    where_mask[..., i, i + 2:] = True

    values = qml.math.zeros_like(matrix)
    values[..., i + 1, i] = alpha
    values[..., i, i + 1] = -alpha
    values[..., i + 2:, i] = zero_like
    values[..., i, i + 2:] = zero_like

    matrix = qml.math.where(where_mask, values, matrix)
    return matrix


def _update_matrix_block_householder(matrix, i, v, tau):
    """
    Update the matrix block A(i+1:N,i+1:N) using the Householder transformation.
    Equivalent to:

        w = tau * A(i+1:N,i+1:N) @ v
        A(i+1:N,i+1:N) -= 2 * v @ w^T

    :param matrix:
    :param i:
    :param v:
    :param tau:
    :return:
    """
    w = tau * qml.math.einsum("...ij,...j->...i", matrix[..., i + 1:, i + 1:], qml.math.conjugate(v))
    values = qml.math.zeros_like(matrix)
    values[..., i + 1:, i + 1:] += (
        qml.math.einsum("...i,...j->...ij", v, w)
        -
        qml.math.einsum("...i,...j->...ij", w, v)
    )
    return matrix + values


def batch_pfaffian_householder(
        __matrix: TensorLike,
        overwrite_input: bool = False,
        test_input: bool = False,
        p_bar: Optional[tqdm.tqdm] = None,
        show_progress: bool = False
):
    """pfaffian(A, overwrite_a=False)

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses the
    Householder tridiagonalization.

    Note that the function pfaffian_schur() can also be used in the
    real case. That function does not make use of the skew-symmetry
    and is only slightly slower than pfaffian_householder().
    """
    if overwrite_input:
        matrix = __matrix
    else:
        matrix = qml.math.ones_like(__matrix) * __matrix
    shape = qml.math.shape(matrix)
    n, m = shape[-2:]
    p_bar = p_bar or tqdm.tqdm(range(n - 2), disable=not show_progress)

    if test_input:
        p_bar.set_description("Testing input matrix")
        # Check if matrix is square
        assert shape[-2] == shape[-1] > 0
        # Check if it's skew-symmetric
        matrix_t = qml.math.einsum("...ij->...ji", matrix)
        assert qml.math.allclose(matrix, -matrix_t)

    matrix = qml.math.cast(matrix, dtype=complex)
    zero_like = convert_and_cast_like(0, matrix)
    pfaffian_val = qml.math.convert_like(np.ones(shape[:-2], dtype=complex), matrix)

    # Quick return if possible
    if n % 2 == 1:
        p_bar.set_description("Odd-sized matrix")
        p_bar.close()
        return pfaffian_val * zero_like * matrix[..., 0, 0]  # 0.0 but with require grad if needed

    p_bar.set_description(f"Computing Pfaffian of {shape} matrix")
    for i in p_bar:
        # Find a Householder vector to eliminate the i-th column
        v, tau, alpha = batch_householder_complex(matrix[..., i + 1:, i])
        matrix = _eliminate_ith_column_householder(matrix, i, alpha)
        matrix = _update_matrix_block_householder(matrix, i, v, tau)

        pfaffian_val = pfaffian_val * qml.math.where(qml.math.isclose(tau, zero_like), 1, 1 - tau)
        if i % 2 == 0:
            pfaffian_val = pfaffian_val * -alpha

    pfaffian_val = pfaffian_val * matrix[..., n - 2, n - 1]
    p_bar.set_description("Pfaffian computed")
    p_bar.close()
    return pfaffian_val


def pfaffian_by_det(
        __matrix: TensorLike,
        p_bar: Optional[tqdm.tqdm] = None,
        show_progress: bool = False,
        epsilon: float = 1e-12
):
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
        pf = qml.math.sqrt(qml.math.abs(det) + epsilon)
    else:
        det = qml.math.det(__matrix)
        pf = qml.math.sqrt(qml.math.abs(det) + epsilon)
    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian_by_det_cuda(
        __matrix: TensorLike,
        p_bar: Optional[tqdm.tqdm] = None,
        show_progress: bool = False,
        epsilon: float = 1e-12
):
    from . import torch_utils
    import torch
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
    pf = qml.math.sqrt(qml.math.abs(det) + epsilon)
    p_bar.set_description(f"Determinant of {shape} matrix computed")
    p_bar.update()
    p_bar.close()
    return pf


def pfaffian(
        __matrix: TensorLike,
        overwrite_input: bool = False,
        method: Literal["P", "H", "det", "bLTL", "bH", "PfaffianFDBPf"] = "bLTL",
        epsilon: float = 1e-12,
        p_bar: Optional[tqdm.tqdm] = None,
        show_progress: bool = False
) -> Union[float, complex, TensorLike]:
    """pfaffian(A, overwrite_a=False, method='P')

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    either the Parlett-Reid algorithm (method='P', default),
    or the Householder tridiagonalization (method='H').

    This code is adapted of the function `pfaffian`
    from https://github.com/basnijholt/pfapack/blob/master/pfapack/pfaffian.py.

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: TensorLike
    :param overwrite_input: Whether to overwrite the input matrix
    :type overwrite_input: bool
    :param method: Method to use. 'P' for Parlett-Reid algorithm, 'H' for Householder tridiagonalization,
        'det' for determinant, 'bLTL' for batched Parlett-Reid algorithm,
        'bH' for batched Householder tridiagonalization,
        'PfaffianFDBPf' for the Pfaffian using the torch_pfaffian library.
    :type method: Literal["P", "H", "det", "bLTL", "bH", "PfaffianFDBPf"]
    :param epsilon: Tolerance for the determinant method
    :type epsilon: float
    :param p_bar: Progress bar
    :type p_bar: Optional[tqdm.tqdm]
    :param show_progress: Whether to show progress bar. If no progress bar is provided, a new one is created
    :type show_progress: bool
    :return: Pfaffian of the matrix
    :rtype: Union[float, complex, TensorLike]
    """
    shape = qml.math.shape(__matrix)
    assert shape[-2] == shape[-1] > 0, "Matrix must be square"

    if method == "P":
        from pfapack.pfaffian import pfaffian
        warnings.warn(
            "The method 'P' is not implemented yet. "
            "It is recommended to use the method 'det' instead.",
            UserWarning,
        )
        return pfaffian(__matrix, overwrite_input, method="P")
    elif method == "H":
        from pfapack.pfaffian import pfaffian_householder
        warnings.warn(
            "The method 'H' is not implemented yet. "
            "It is recommended to use the method 'det' instead.",
            UserWarning,
        )
        return pfaffian_householder(__matrix, overwrite_input)
    elif method == "det":
        return pfaffian_by_det(__matrix, p_bar=p_bar, show_progress=show_progress, epsilon=epsilon)
    elif method == "cuda_det":
        return pfaffian_by_det_cuda(__matrix, p_bar=p_bar, show_progress=show_progress, epsilon=epsilon)
    elif method == "bLTL":
        return batch_pfaffian_ltl(__matrix, overwrite_input, show_progress=show_progress, p_bar=p_bar)
    elif method == "bH":
        return batch_pfaffian_householder(__matrix, overwrite_input, show_progress=show_progress, p_bar=p_bar)
    elif method == "PfaffianFDBPf":
        try:
            import torch_pfaffian
        except ImportError:
            raise ImportError("torch_pfaffian is not installed."
                              "Please install it with `pip install https://github.com/MatchCake/TorchPfaffian.git`.")
        from . import torch_utils
        pf = torch_pfaffian.get_pfaffian_function(method)(torch_utils.to_tensor(__matrix, dtype=torch.complex128))
        return convert_and_cast_like(pf, __matrix)
    else:
        raise ValueError(f"Invalid method. Got {method}, must be 'P', 'H', 'det', 'bLTL', or 'bH'.")
