from typing import Union
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import warnings
from .math import convert_and_cast_like


def pfaffian_ltl(__matrix, overwrite_input=False) -> Union[float, complex]:
    r"""
    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.

    This code is adapted of the function `pfaffian_LTL`
    from https://github.com/basnijholt/pfapack/blob/master/pfapack/pfaffian.py.

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: np.ndarray
    :param overwrite_input: Whether to overwrite the input matrix
    :type overwrite_input: bool
    :return: Pfaffian of the matrix
    :rtype: Union[float, complex]
    """
    if overwrite_input:
        matrix = __matrix
    else:
        matrix = qml.math.ones_like(__matrix) * __matrix
    shape = qml.math.shape(matrix)
    # Check if matrix is square
    assert shape[-2] == shape[-1] > 0
    # Check if it's skew-symmetric
    matrix_t = qml.math.einsum("...ij->...ji", matrix)
    assert qml.math.allclose(matrix, -matrix_t)

    n, m = shape[-2:]
    # Quick return if possible
    if n % 2 == 1:
        return 0.0
    matrix = qml.math.cast(matrix, dtype=complex)
    zero_like = convert_and_cast_like(0, matrix)
    pfaffian_val = qml.math.convert_like(pnp.ones(shape[:-2], dtype=complex), matrix)
    
    for k in range(0, n - 1, 2):
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + qml.math.abs(matrix[..., k + 1:, k]).argmax(axis=-1)
        
        # Check if we need to pivot
        if kp != k + 1:  # TODO: need to add batch support
            # interchange rows k+1 and kp
            temp = qml.math.ones_like(matrix[..., k + 1, k:]) * matrix[..., k + 1, k:]
            matrix[..., k + 1, k:] = matrix[..., kp, k:]
            matrix[..., kp, k:] = temp
            
            # Then interchange columns k+1 and kp
            temp = qml.math.ones_like(matrix[..., k:, k + 1]) * matrix[..., k:, k + 1]
            matrix[..., k:, k + 1] = matrix[..., k:, kp]
            matrix[..., k:, kp] = temp
            
            # every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -1
        
        if qml.math.isclose(matrix[..., k + 1, k], zero_like):
            # if we encounter a zero on the super/subdiagonal, the Pfaffian is 0
            return zero_like
        else:
            # Now form the Gauss vector
            tau = qml.math.ones_like(matrix[..., k, k + 2:]) * matrix[..., k, k + 2:]
            zero_mask = qml.math.isclose(matrix[..., k, k + 1], zero_like)
            tau = tau / matrix[..., k, k + 1]
            tau = qml.math.where(zero_mask, zero_like, tau)
            pfaffian_val *= matrix[..., k, k + 1]
            
            if k + 2 < n:
                # Update the matrix block A(k+2:,k+2)
                matrix[..., k + 2:, k + 2:] = matrix[..., k + 2:, k + 2:] + qml.math.outer(
                    tau, matrix[..., k + 2:, k + 1]
                )
                matrix[..., k + 2:, k + 2:] = matrix[..., k + 2:, k + 2:] - qml.math.outer(
                    matrix[..., k + 2:, k + 1], tau
                )
    
    return pfaffian_val


def _pivot(__matrix, k, kp):
    matrix = qml.math.ones_like(__matrix) * __matrix
    kp1 = np.full_like(kp, k, dtype=int) + 1
    # interchange rows k+1 and kp
    temp = qml.math.ones_like(matrix[..., kp1, k:]) * matrix[..., kp1, k:]
    matrix[..., kp1, k:] = matrix[..., kp, k:]
    matrix[..., kp, k:] = temp

    # Then interchange columns k+1 and kp
    temp = qml.math.ones_like(matrix[..., k:, kp1]) * matrix[..., k:, kp1]
    matrix[..., k:, kp1] = matrix[..., k:, kp]
    matrix[..., k:, kp] = temp
    return matrix


def _batch_pfaffian_ltl(__matrix, overwrite_input=False, test_input: bool = False) -> Union[float, complex]:
    r"""
    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.

    This code is adapted of the function `pfaffian_LTL`
    from https://github.com/basnijholt/pfapack/blob/master/pfapack/pfaffian.py.

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: np.ndarray
    :param overwrite_input: Whether to overwrite the input matrix
    :type overwrite_input: bool
    :return: Pfaffian of the matrix
    :rtype: Union[float, complex]
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
    if len(shape) > 2:
        batch_size = shape[0]
    else:
        batch_size = 1
    matrix = qml.math.cast(matrix, dtype=complex)
    zero_like = convert_and_cast_like(0, matrix)
    pfaffian_val = qml.math.convert_like(pnp.ones(shape[:-2], dtype=complex), matrix)

    # Quick return if possible
    if n % 2 == 1:
        return qml.math.zeros_like(pfaffian_val)

    for k in range(0, n - 1, 2):
        # kv = np.full(batch_size, k, dtype=int)
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + qml.math.abs(matrix[..., k + 1:, k]).argmax(-1)

        # Check if we need to pivot
        pivot_condition = ~qml.math.isclose(kp, k + 1)
        # interchange rows and cols k+1 and kp (pivot if needed)
        matrix = qml.math.where(pivot_condition[..., None, None], _pivot(matrix, k, kp), matrix)
        # every interchange corresponds to a "-" in det(P)
        pfaffian_val *= qml.math.where(pivot_condition, -1, 1)

        # if we encounter a zero on the super/subdiagonal, the Pfaffian is 0
        zero_ss_condition = qml.math.isclose(matrix[..., k + 1, k], zero_like)
        pfaffian_val *= qml.math.where(zero_ss_condition, 0.0, 1.0)
        if qml.math.all(zero_ss_condition):
            return pfaffian_val

        # Now form the Gauss vector
        tau_norm = matrix[..., k, k + 1][..., None]
        zero_mask = qml.math.isclose(tau_norm, zero_like)
        tau = qml.math.where(zero_mask, zero_like, matrix[..., k, k + 2:] / tau_norm)
        pfaffian_val *= matrix[..., k, k + 1]

        if k + 2 < n:
            # Update the matrix block A(k+2:,k+2)
            matrix[..., k + 2:, k + 2:] += qml.math.einsum("...i,...j->...ij", tau, matrix[..., k + 2:, k + 1])
            matrix[..., k + 2:, k + 2:] -= qml.math.einsum("...i,...j->...ij", matrix[..., k + 2:, k + 1], tau)

    return pfaffian_val


def batch_pfaffian_ltl(__matrix, overwrite_input=False) -> Union[float, complex]:
    ndim = qml.math.ndim(__matrix)
    if ndim == 2:
        return pfaffian_ltl(__matrix, overwrite_input)
    elif ndim == 3:
        # TODO: need to add batch support
        return qml.math.stack([
            pfaffian_ltl(__matrix[i], overwrite_input)
            for i in range(qml.math.shape(__matrix)[0])
        ])
    else:
        raise ValueError(f"Invalid ndim. Got {ndim}, must be 2 or 3.")


def pfaffian(__matrix, overwrite_input=False, method="P") -> Union[float, complex]:
    """pfaffian(A, overwrite_a=False, method='P')

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    either the Parlett-Reid algorithm (method='P', default),
    or the Householder tridiagonalization (method='H').

    This code is adapted of the function `pfaffian`
    from https://github.com/basnijholt/pfapack/blob/master/pfapack/pfaffian.py.

    :param __matrix: Matrix to compute the Pfaffian of
    :type __matrix: np.ndarray
    :param overwrite_input: Whether to overwrite the input matrix
    :type overwrite_input: bool
    :param method: Method to use. Either 'P' or 'H'.
    :type method: str
    :return: Pfaffian of the matrix
    :rtype: Union[float, complex]
    """
    shape = qml.math.shape(__matrix)
    assert shape[-2] == shape[-1] > 0
    
    if method == "P":
        return batch_pfaffian_ltl(__matrix, overwrite_input)
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
            return qml.math.sqrt(qml.math.abs(qml.math.linalg.det(__matrix)))
        return qml.math.sqrt(qml.math.abs(qml.math.det(__matrix)))
    elif method == "bLTL":
        return _batch_pfaffian_ltl(__matrix, overwrite_input)
    else:
        raise ValueError(f"Invalid method. Got {method}, must be 'P' or 'H'.")
