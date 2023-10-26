import importlib
from typing import Any, List, Callable

import numpy as np


PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULI_I = np.eye(2)


def recursive_kron(__inputs: List[Any], lib=np) -> Any:
    if isinstance(lib, str):
        lib = importlib.import_module(lib)
    return recursive_2in_operator(lib.kron, __inputs)

    
def recursive_2in_operator(operator: Callable, __inputs: List[Any]):
    if len(__inputs) == 1:
        return __inputs[0]
    elif len(__inputs) == 2:
        return operator(__inputs[0], __inputs[1])
    elif len(__inputs) > 2:
        rec = recursive_2in_operator(operator, __inputs[:-1])
        return operator(rec, __inputs[-1])
    else:
        raise ValueError("Invalid shape for input array")


def get_majorana_pauli_list(i: int, n: int) -> List[np.ndarray]:
    r"""

    Get the list of Pauli matrices for the computation of the Majorana operator :math:`c_i` defined as

    .. math::
        c_{2k+1} = Z^{\otimes k} \otimes X \otimes I^{\otimes n-k-1}

    for odd :math:`i` and

    .. math::
        c_{2k} = Z^{\otimes k} \otimes Y \otimes I^{\otimes n-k-1}

    for even :math:`i`, where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix, :math:`X`
    is the Pauli X matrix, :math:`\otimes` is the Kronecker product, :math:`k` is the index of the Majorana
    operator and :math:`n` is the number of particles.

    :param i: Index of the Majorana operator
    :type i: int
    :param n: Number of particles
    :type n: int
    :return: List of Pauli matrices
    :rtype: List[np.ndarray]
    """
    assert 0 <= i < 2 * n
    k = int(i / 2)  # 0, ..., n-1
    if (i + 1) % 2 == 0:
        gate = PAULI_Y
    else:
        gate = PAULI_X
    return [PAULI_Z] * k + [gate] + [PAULI_I] * (n - k - 1)


def get_majorana_pauli_string(i: int, n: int) -> str:
    assert 0 <= i < 2 * n
    k = int(i / 2)  # 0, ..., n-1
    if (i + 1) % 2 == 0:
        gate = "Y"
    else:
        gate = "X"
    return '⊗'.join(["Z"] * k + [gate] + ["I"] * (n - k - 1))


def get_majorana(i: int, n: int) -> np.ndarray:
    r"""
    Get the Majorana matrix defined as

    .. math::
        c_{2k+1} = Z^{\otimes k} \otimes X \otimes I^{\otimes n-k-1}

    for odd :math:`i` and

    .. math::
        c_{2k} = Z^{\otimes k} \otimes Y \otimes I^{\otimes n-k-1}

    for even :math:`i`, where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix, :math:`X`
    is the Pauli X matrix, :math:`\otimes` is the Kronecker product, :math:`k` is the index of the Majorana
    operator and :math:`n` is the number of particles.

    :Note: The index :math:`i` starts from 0.

    :param i: Index of the Majorana operator
    :type i: int
    :param n: Number of particles
    :type n: int
    :return: Majorana matrix
    :rtype: np.ndarray
    """
    return recursive_kron(get_majorana_pauli_list(i, n))


def get_non_interacting_fermionic_hamiltonian_from_coeffs(hamiltonian_coefficients_matrix, lib=np):
    r"""
    Compute the non-interacting fermionic Hamiltonian from the coefficients of the Majorana operators.

    .. math::
        H = i \sum_{\mu\neq\nu}^{2n} h_{\mu \nu} c_\mu c_\nu

    where :math:`h_{\mu \nu}` are the coefficients of the Majorana operators :math:`c_\mu` and :math:`c_\nu`,
    :math:`n` is the number of particles, :math:`\mu` and :math:`\nu` are the indices of the Majorana operators, and
    :math:`i` is the imaginary unit.

    TODO: optimize the method by changing the sum for a matrix multiplication as :math:`H = i C^T h C` where :math:`C`
        is the matrix of Majorana operators.
        
    TODO: use multiprocessing to parallelize the computation of the matrix elements.

    :param hamiltonian_coefficients_matrix: Coefficients of the Majorana operators. Must be a square matrix of shape
        :math:`(2n, 2n)`.
    :param lib: Library to use for the operations
    :return: Non-interacting fermionic Hamiltonian
    """
    # if isinstance(lib, str):
    #     lib = eval(lib)

    n_particles = int(len(hamiltonian_coefficients_matrix) / 2)
    hamiltonian = np.zeros((2**n_particles, 2**n_particles), dtype=complex)

    for mu in range(2*n_particles):
        for nu in range(2*n_particles):
            c_mu = get_majorana(mu, n_particles)
            c_nu = get_majorana(nu, n_particles)
            hamiltonian += hamiltonian_coefficients_matrix[mu, nu] * c_mu @ c_nu
    return hamiltonian


def skew_antisymmetric_vector_to_matrix(__vector) -> np.ndarray:
    r"""

    Compute the skew-antisymmetric matrix from a vector. The skew-antisymmetric (NxN) matrix is defined as

    .. math::
        \mathbf{A} =
        \begin{pmatrix}
            0 & a_0 & a_1 & a_2 & \dots & a_{N-1} \\
            -a_0 & 0 & a_{N} & a_{N+1} & \dots & a_{2N-2} \\
            -a_1 & -a_{N} & 0 & a_{2N} & \dots & a_{3N-3} \\
            -a_2 & -a_{N+1} & -a_{2N} & 0 & \dots & a_{4N-4} \\
            \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
            -a_{N-1} & -a_{2N-2} & -a_{3N-3} & -a_{4N-4} & \dots & 0
        \end{pmatrix}

    where :math:`a_i` are the elements of the vector :math:`\mathbf{a}` of length :math:`N(N-1)/2`.

    :Note: The length of the vector must be :math:`N(N-1)/2` where :math:`(N, N)` is the shape of the matrix.

    :param __vector: Vector of length :math:`N(N-1)/2`
    :return: Skew-antisymmetric matrix of shape :math:`(N, N)`
    """
    vector_1d = np.asarray(__vector).flatten()
    ell = len(vector_1d)
    n = int(0.5 * (1 + np.sqrt(1 + 8 * ell)))
    if n * (n - 1) / 2 != ell:
        raise ValueError(f"Invalid vector length: {ell}")
    matrix = np.zeros((n, n))
    current_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            matrix[i, j] = vector_1d[current_idx]
            current_idx += 1
    matrix -= matrix.T
    return matrix


def check_if_imag_is_zero(__matrix: np.ndarray, eps: float = 1e-5) -> bool:
    r"""

    Check if the imaginary part of a matrix is zero.

    :param __matrix: Matrix to check
    :param eps: Tolerance for the imaginary part
    :return: True if the imaginary part is zero, False otherwise
    """
    return np.allclose(__matrix.imag, 0.0, atol=eps)


def make_transition_matrix_from_action_matrix(action_matrix):
    r"""
    
    Compute the transition matrix from the action matrix. The transition matrix is defined as
    :math:`\mathbf{T}` such that
    
    .. math::
        \mathbf{T}_{i,\nu} = \frac{1}{2} \left( \mathbf{A}^T_{2i-1,\nu} + i \mathbf{A}^T_{2i,\nu} \right)
    
    where :math:`\mathbf{A}` is the action matrix of shape (2n x 2n), :math:`\mathbf{T}` is the transition matrix
    of shape (n x 2n), :math:`i` goes from 1 to :math:`n` and :math:`\nu` goes from 1 to :math:`2n`.
    
    :param action_matrix:
    :return:
    """
    n = action_matrix.shape[0] // 2
    transition_matrix = 0.5 * (
            action_matrix.T[0:n - 1:2] + 1j * action_matrix.T[1:n:2]
    )
    return transition_matrix


def get_block_diagonal_matrix(n: int) -> np.ndarray:
    r"""
    Construct the special block diagonal matrix of shape (2n x 2n) defined as
    
    .. math::
        \mathbf{B} =
        \oplus_{j=1}^{n}
        \begin{pmatrix}
            1 & i \\
            -i & 1
        \end{pmatrix}
    
    where :math:`\oplus` is the direct sum operator, :math:`n` is the number of particles and :math:`i` is the
    imaginary unit.
    
    :param n: Number of particles
    :type n: int
    :return: Block diagonal matrix of shape (2n x 2n)
    """
    block_diagonal_matrix = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        block_diagonal_matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.array([[1, 1j], [-1j, 1]])
    return block_diagonal_matrix

