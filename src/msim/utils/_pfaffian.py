from typing import Union
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import warnings


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
        matrix = __matrix.copy()
    shape = qml.math.shape(matrix)
    # Check if matrix is square
    assert shape[-2] == shape[-1] > 0
    # Check if it's skew-symmetric
    matrix_t = qml.math.swapaxes(matrix, -2, -1)
    assert qml.math.allclose(matrix, -matrix_t)
    
    n, m = shape[-2:]
    # Quick return if possible
    if n % 2 == 1:
        return 0.0
    matrix = matrix.astype(np.complex128)
    pfaffian_val = pnp.ones(shape[:-2], dtype=complex)
    
    for k in range(0, n - 1, 2):
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + qml.math.abs(matrix[..., k + 1:, k]).argmax(axis=-1)
        
        # Check if we need to pivot
        if kp != k + 1:  # TODO: need to add batched support
            # interchange rows k+1 and kp
            temp = matrix[..., k + 1, k:].copy()
            matrix[..., k + 1, k:] = matrix[..., kp, k:]
            matrix[..., kp, k:] = temp
            
            # Then interchange columns k+1 and kp
            temp = matrix[..., k:, k + 1].copy()
            matrix[..., k:, k + 1] = matrix[..., k:, kp]
            matrix[..., k:, kp] = temp
            
            # every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -1
        
        if np.isclose(matrix[..., k + 1, k], 0.0):
            # if we encounter a zero on the super/subdiagonal, the Pfaffian is 0
            return 0.0
        else:
            # Now form the Gauss vector
            tau = matrix[..., k, k + 2:].copy()
            tau = pnp.divide(
                tau, matrix[..., k, k + 1],
                out=pnp.zeros_like(tau),
                where=not np.isclose(matrix[..., k, k + 1], 0.0)
            )
            pfaffian_val *= matrix[..., k, k + 1]
            
            if k + 2 < n:
                # Update the matrix block A(k+2:,k+2)
                matrix[..., k + 2:, k + 2:] = matrix[..., k + 2:, k + 2:] + np.outer(
                    tau, matrix[..., k + 2:, k + 1]
                )
                matrix[..., k + 2:, k + 2:] = matrix[..., k + 2:, k + 2:] - np.outer(
                    matrix[..., k + 2:, k + 1], tau
                )
    
    return pfaffian_val


def batch_pfaffian_ltl(__matrix, overwrite_input=False) -> Union[float, complex]:
    ndim = qml.math.ndim(__matrix)
    if ndim == 2:
        return pfaffian_ltl(__matrix, overwrite_input)
    elif ndim == 3:
        return qml.math.stack(
            [pfaffian_ltl(__matrix[i], overwrite_input) for i in range(__matrix.shape[0])]
        )
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
            "The method 'H' is not implemented in PennyLane. "
            "It is recommended to use the method 'P' instead.",
            UserWarning,
        )
        return pfaffian_householder(__matrix, overwrite_input)
    else:
        raise ValueError(f"Invalid method. Got {method}, must be 'P' or 'H'.")
