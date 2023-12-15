from typing import Union

import numpy as np


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
    # Check if matrix is square
    assert matrix.shape[0] == matrix.shape[1] > 0
    # Check if it's skew-symmetric
    assert np.abs((matrix + matrix.T).max()) < 1e-14
    
    n, m = matrix.shape
    # Quick return if possible
    if n % 2 == 1:
        return 0.0
    matrix = matrix.astype(np.complex128)
    pfaffian_val = 1.0
    
    for k in range(0, n - 1, 2):
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + np.abs(matrix[k + 1:, k]).argmax()
        
        # Check if we need to pivot
        if kp != k + 1:
            # interchange rows k+1 and kp
            temp = matrix[k + 1, k:].copy()
            matrix[k + 1, k:] = matrix[kp, k:]
            matrix[kp, k:] = temp
            
            # Then interchange columns k+1 and kp
            temp = matrix[k:, k + 1].copy()
            matrix[k:, k + 1] = matrix[k:, kp]
            matrix[k:, kp] = temp
            
            # every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -1
        
        if np.isclose(matrix[k + 1, k], 0.0):
            # if we encounter a zero on the super/subdiagonal, the Pfaffian is 0
            return 0.0
        else:
            # Now form the Gauss vector
            tau = matrix[k, k + 2:].copy()
            tau = np.divide(
                tau, matrix[k, k + 1],
                out=np.zeros_like(tau),
                where=not np.isclose(matrix[k, k + 1], 0.0)
            )
            pfaffian_val *= matrix[k, k + 1]
            
            if k + 2 < n:
                # Update the matrix block A(k+2:,k+2)
                matrix[k + 2:, k + 2:] = matrix[k + 2:, k + 2:] + np.outer(
                    tau, matrix[k + 2:, k + 1]
                )
                matrix[k + 2:, k + 2:] = matrix[k + 2:, k + 2:] - np.outer(
                    matrix[k + 2:, k + 1], tau
                )
    
    return pfaffian_val


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
    # Check if matrix is square
    assert __matrix.shape[0] == __matrix.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((__matrix + __matrix.T).max()) < 1e-14
    
    if method == "P":
        return pfaffian_ltl(__matrix, overwrite_input)
    elif method == "H":
        from pfapack.pfaffian import pfaffian_householder
        return pfaffian_householder(__matrix, overwrite_input)
    else:
        raise ValueError(f"Invalid method. Got {method}, must be 'P' or 'H'.")
