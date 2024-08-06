from typing import Union, Literal
import pennylane as qml
import numpy as np
import warnings
from .math import convert_and_cast_like
from ..templates import TensorLike


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
    add_subb_matrix = (
        qml.math.einsum("...i,...j->...ij", tau, __matrix[..., k + 2:, k + 1])
        -
        qml.math.einsum("...i,...j->...ij", __matrix[..., k + 2:, k + 1], tau)
    )
    add_matrix[..., k + 2:, k + 2:] += add_subb_matrix
    return __matrix + add_matrix


def batch_pfaffian_ltl(
        __matrix: TensorLike,
        overwrite_input: bool = False,
        test_input: bool = False
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

    :return: Pfaffian of the matrix
    :rtype: Union[float, complex, TensorLike]
    """
    if overwrite_input:
        matrix = __matrix
    else:
        matrix = qml.math.ones_like(__matrix) * __matrix
    shape = qml.math.shape(matrix)
    if test_input:
        # Check if matrix is square
        assert shape[-2] == shape[-1] > 0
        # Check if it's skew-symmetric
        matrix_t = qml.math.einsum("...ij->...ji", matrix)
        assert qml.math.allclose(matrix, -matrix_t)

    n, m = shape[-2:]
    matrix = qml.math.cast(matrix, dtype=complex)
    zero_like = convert_and_cast_like(0, matrix)
    pfaffian_val = qml.math.convert_like(np.ones(shape[:-2], dtype=complex), matrix)

    # Quick return if possible
    if n % 2 == 1:
        return pfaffian_val * zero_like * matrix[..., 0, 0]  # 0.0 but with require grad if needed

    for k in range(0, n - 1, 2):
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
            return pfaffian_val

        if k + 2 < n:
            tau = _compute_gauss_vector(matrix, k)
            # Update the matrix block A(k+2:,k+2)
            matrix = _update_matrix_block_kp2_kp2(matrix, k, tau)
    return pfaffian_val


def pfaffian(
        __matrix: TensorLike,
        overwrite_input: bool = False,
        method: Literal["P", "H", "det", "bLTL"] = "bLTL",
        epsilon: float = 1e-12
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
        'det' for determinant, 'bLTL' for batched Parlett-Reid algorithm
    :type method: Literal["P", "H", "det", "bLTL"]
    :param epsilon: Tolerance for the determinant method
    :type epsilon: float
    :return: Pfaffian of the matrix
    :rtype: Union[float, complex, TensorLike]
    """
    shape = qml.math.shape(__matrix)
    assert shape[-2] == shape[-1] > 0
    
    if method == "P":
        from pfapack.pfaffian import pfaffian
        warnings.warn(
            "The method 'H' is not implemented yet. "
            "It is recommended to use the method 'P' instead.",
            UserWarning,
        )
        return pfaffian(__matrix, overwrite_input, method="P")
    elif method == "H":
        from pfapack.pfaffian import pfaffian_householder
        warnings.warn(
            "The method 'H' is not implemented yet. "
            "It is recommended to use the method 'P' instead.",
            UserWarning,
        )
        return pfaffian_householder(__matrix, overwrite_input)
    elif method == "det":
        backend = qml.math.get_interface(__matrix)
        if backend in ["autograd", "numpy"]:
            return qml.math.sqrt(qml.math.abs(qml.math.linalg.det(__matrix)) + epsilon)
        return qml.math.sqrt(qml.math.abs(qml.math.det(__matrix)) + epsilon)
    elif method == "bLTL":
        return batch_pfaffian_ltl(__matrix, overwrite_input)
    else:
        raise ValueError(f"Invalid method. Got {method}, must be 'P' or 'H'.")
