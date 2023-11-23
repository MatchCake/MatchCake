import importlib
from typing import Any, List, Callable, Union

import numpy as np
import pennylane as qml
from pennylane.wires import Wires

PAULI_X = np.array([
    [0, 1],
    [1, 0]
])
PAULI_Y = np.array([
    [0, -1j],
    [1j, 0]
])
PAULI_Z = np.array([
    [1, 0],
    [0, -1]
])
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


def binary_string_to_vector(binary_string: str, encoding: str = "ascii") -> np.ndarray:
    r"""
    Convert a binary string to a vector. The binary string is a string of 0s and 1s. The vector is a vector of
    integers.
    
    :param binary_string: Binary string
    :type binary_string: str
    :param encoding: Encoding of the binary string. Default is ascii.
    :type encoding: str
    :return: Vector
    """
    return np.frombuffer(binary_string.encode(encoding), dtype='u1') - ord('0')


def binary_state_to_state(binary_state: Union[np.ndarray, List[Union[int, bool]], str]) -> np.ndarray:
    r"""
    Convert a binary state to a state. The binary state is binary string of length :math:`2^n` where :math:`n` is
    the number of particles. The state is a vector of length :math:`2^n` where :math:`n` is the number of particles.

    :param binary_state: Binary state
    :type binary_state: Union[np.ndarray, List[Union[int, bool]], str]
    :return: State
    :rtype: np.ndarray
    """
    if isinstance(binary_state, str):
        binary_state = binary_string_to_vector(binary_state)
    elif isinstance(binary_state, list):
        binary_state = np.asarray(binary_state, dtype=int).flatten()
    states = [np.asarray([1-s, s]) for s in binary_state]
    state = recursive_kron(states)
    return state


def state_to_binary_state(state: np.ndarray) -> str:
    r"""
    Convert a state to a binary state. The binary state is binary string of length :math:`2^n` where :math:`n` is
    the number of particles. The state is a vector of length :math:`2^n` where :math:`n` is the number of particles.

    :param state: State
    :type state: np.ndarray
    :return: Binary state
    :rtype: str
    """
    state = np.asarray(state).flatten()
    n_states = state.shape[0]
    n = int(np.log2(n_states))
    assert n_states == 2 ** n, f"Invalid number of states: {n_states}, must be a power of 2."
    state_number = np.argmax(state)
    binary_state = np.binary_repr(state_number, width=n)
    return binary_state


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
    # return [PAULI_Z] * k + [gate] + [PAULI_I] * (n - k - 1)
    return [PAULI_Z for _ in range(k)] + [gate] + [PAULI_I for _ in range(n - k - 1)]


def get_majorana_pauli_string(i: int, n: int, join_char='⊗') -> str:
    assert 0 <= i < 2 * n
    k = int(i / 2)  # 0, ..., n-1
    if (i + 1) % 2 == 0:
        gate = "Y"
    else:
        gate = "X"
    return join_char.join(["Z"] * k + [gate] + ["I"] * (n - k - 1))


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


def get_non_interacting_fermionic_hamiltonian_from_coeffs(
        hamiltonian_coefficients_matrix,
        energy_offset=0.0,
        lib=np
):
    r"""
    Compute the non-interacting fermionic Hamiltonian from the coefficients of the Majorana operators.

    .. math::
        H = -i\sum_{\mu,\nu = 0}^{2n-1} h_{\mu \nu} c_\mu c_\nu + \epsilon \mathbb{I}

    where :math:`h_{\mu \nu}` are the coefficients of the Majorana operators :math:`c_\mu` and :math:`c_\nu`,
    :math:`n` is the number of particles, :math:`\mu`, :math:`\nu` are the indices of the Majorana operators,
    :math:`\epsilon` is the energy offset and :math:`\mathbb{I}` is the identity matrix.

    TODO: optimize the method by changing the sum for a matrix multiplication as :math:`H = i C^T h C` where :math:`C`
        is the matrix of Majorana operators.
        
    TODO: use multiprocessing to parallelize the computation of the matrix elements.

    :param hamiltonian_coefficients_matrix: Coefficients of the Majorana operators. Must be a square matrix of shape
        :math:`(2n, 2n)`.
    :type hamiltonian_coefficients_matrix: np.ndarray
    :param energy_offset: Energy offset
    :type energy_offset: float
    :param lib: Library to use for the operations
    :return: Non-interacting fermionic Hamiltonian
    """
    backend = load_backend_lib(lib)
    n_particles = int(len(hamiltonian_coefficients_matrix) / 2)
    hamiltonian = energy_offset * backend.eye(2**n_particles, dtype=complex)

    for mu in range(2*n_particles):
        for nu in range(2*n_particles):
            c_mu = get_majorana(mu, n_particles)
            c_nu = get_majorana(nu, n_particles)
            hamiltonian += -1j * hamiltonian_coefficients_matrix[mu, nu] * (c_mu @ c_nu)
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


def decompose_matrix_into_majoranas(__matrix: np.ndarray) -> np.ndarray:
    r"""
    Decompose a matrix into Majorana operators. The matrix is decomposed as

    .. math::
        \mathbf{M} = \sum_{i=0}^{2^{n}-1} m_i c_i

    where :math:`\mathbf{M}` is the matrix, :math:`m_i` are the coefficients of the matrix, :math:`n` is the number
    of particles and :math:`c_i` are the Majorana operators.

    :param __matrix: Input matrix
    :type __matrix: np.ndarray
    :return: Coefficients of the Majorana operators
    :rtype: np.ndarray
    """
    n_states = __matrix.shape[0]
    n = int(np.log2(n_states))
    assert n_states == 2 ** n, f"Invalid number of states: {n_states}, must be a power of 2."
    assert n == 2, f"Invalid number of particles: {n}, must be 2."
    assert __matrix.shape == (n_states, n_states), f"Invalid shape for matrix: {__matrix.shape}, must be square."
    matrix = __matrix.copy()
    matrix = matrix.astype(complex)
    coeffs = np.zeros(n_states, dtype=complex)
    for i in range(n_states):
        c_i = get_majorana(i, n)
        coeffs[i] = np.trace(matrix @ c_i) / n_states
    return coeffs


def decompose_state_into_majorana_indexes(__state: np.ndarray) -> np.ndarray:
    r"""
    Decompose a state into Majorana operators. The state is decomposed as

    .. math::
        |x> = c_{2p_{1}} ... c_{2p_{\ell}} |0>

    where :math:`|x>` is the state, :math:`c_i` are the Majorana operators, :math:`p_i` are the indices of the
    Majorana operators and :math:`\ell` is the hamming weight of the state.

    Note: The state must be a pure state in the computational basis.

    :param __state: Input state
    :type __state: np.ndarray
    :return: Indices of the Majorana operators
    :rtype: np.ndarray
    """
    binary_state = state_to_binary_state(__state)
    return decompose_binary_state_into_majorana_indexes(binary_state)


def decompose_binary_state_into_majorana_indexes(
        __binary_state: Union[np.ndarray, List[Union[int, bool]], str]
) -> np.ndarray:
    r"""
    Decompose a state into Majorana operators. The state is decomposed as

    .. math::
        |x> = c_{2p_{1}} ... c_{2p_{\ell}} |0>

    where :math:`|x>` is the state, :math:`c_i` are the Majorana operators, :math:`p_i` are the indices of the
    Majorana operators and :math:`\ell` is the hamming weight of the state.

    Note: The state must be a pure state in the computational basis.

    :param __binary_state: Input state as a binary string.
    :type __binary_state: Union[np.ndarray, List[Union[int, bool]], str]
    :return: Indices of the Majorana operators
    :rtype: np.ndarray
    """
    if isinstance(__binary_state, str):
        binary_state_array = binary_string_to_vector(__binary_state)
    else:
        binary_state_array = np.asarray(__binary_state, dtype=int).flatten()
    majorana_indexes = 2 * np.nonzero(binary_state_array)[0]
    return majorana_indexes


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
    transition_matrix = 0.5 * (
            action_matrix.T[::2] + 1j * action_matrix.T[1::2]
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


def get_hamming_weight(state: np.ndarray) -> int:
    r"""
    
    Compute the Hamming weight of a state. The Hamming weight is defined as the number of non-zero elements in the
    state.
    
    The binary state is a one hot vector of shape (2^n,) where n is the number of particles.
    The Hamming weight is the number of states in the state [0, 1].
    
    :param state: State of the system
    :type state: np.ndarray
    :return: Hamming weight of the state
    :rtype: int
    """
    state_reshape = state.reshape(-1, 2)
    hamming_weight = np.sum(state_reshape[:, 1], dtype=int)
    return hamming_weight


def get_4x4_non_interacting_fermionic_hamiltonian_from_params(params):
    r"""

    Compute the non-interacting fermionic Hamiltonian from the parameters of the Matchgate model.

    :param params: Parameters of the Matchgate model
    :type params: MatchgateParams
    :return: Non-interacting fermionic Hamiltonian
    :rtype: np.ndarray
    """
    from .matchgate_parameter_sets import MatchgateHamiltonianCoefficientsParams
    params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
    return np.array([
        [-2j * (params.h0 + params.h5), 0, 0, 2*(params.h4 - params.h1) + 2j*(params.h2 + params.h3)],
        [0, 2j * (params.h0 - params.h5), 2j * (params.h3 - params.h2) - 2*(params.h1 + params.h4), 0],
        [0, 2 * (params.h1 + params.h4) + 2j*(params.h3 - params.h2), 2j * (params.h5 - params.h0), 0],
        [2*(params.h1 - params.h4) + 2j*(params.h2 + params.h3), 0, 0, -2j * (params.h0 + params.h5)],
    ], dtype=complex)


def get_unitary_from_hermitian_matrix(matrix: np.ndarray) -> np.ndarray:
    r"""
    Get the unitary matrix from a Hermitian matrix. The unitary matrix is defined as

    .. math::
        U = e^{-iH}

    where :math:`H` is the Hermitian matrix and :math:`i` is the imaginary unit.

    :param matrix: Hermitian matrix
    :type matrix: np.ndarray
    :return: Unitary matrix
    :rtype: np.ndarray
    """
    from scipy.linalg import expm
    return expm(1j * matrix.astype(complex))


def cast_to_complex(__inputs):
    r"""

    Cast the inputs to complex numbers.

    :param __inputs: Inputs to cast
    :return: Inputs casted to complex numbers
    """
    return type(__inputs)(np.asarray(__inputs).astype(complex))


def load_backend_lib(backend):
    if isinstance(backend, str):
        backend = importlib.import_module(backend)
    return backend


def camel_case_to_spaced_camel_case(__string: str) -> str:
    r"""

    Convert a camel case string to a spaced camel case string. The conversion is done by adding a space before
    every capital letter.

    :param __string: Camel case string
    :type __string: str
    :return: Spaced camel case string
    :rtype: str
    """
    spaced_camel_case_string = ""
    for i, char in enumerate(__string):
        if char.isupper() and i > 0:
            spaced_camel_case_string += " "
        spaced_camel_case_string += char
    return spaced_camel_case_string


def get_probabilities_from_state(state: np.ndarray, wires=None) -> np.ndarray:
    r"""
    
    Compute the probabilities from a state. The probabilities are defined as
    
    .. math::
        p_i = |x_i|^2
    
    where :math:`|x_i>` is the state.
    
    :param state: State of the system
    :type state: np.ndarray
    :param wires: Wires to consider
    :type wires: list[int]
    :return: Probabilities
    :rtype: np.ndarray
    """
    n_states = int(np.log2(len(state)))
    all_wires = Wires(list(range(n_states)))
    if wires is None:
        wires = all_wires
    elif isinstance(wires, int):
        wires = [wires]
        wires = all_wires.subset(wires)
    else:
        wires = all_wires.subset(wires)
    meas = qml.measurements.ProbabilityMP(wires=wires)
    return meas.process_state(state=state, wire_order=all_wires)
