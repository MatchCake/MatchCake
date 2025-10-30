import importlib
from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from scipy import sparse

from . import constants, cuda, majorana, math, operators, torch_utils
from ._pfaffian import pfaffian
from .constants import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from .majorana import (
    MajoranaGetter,
    get_majorana,
    get_majorana_pauli_list,
    get_majorana_pauli_string,
)
from .operators import recursive_2in_operator, recursive_kron


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
    return np.frombuffer(binary_string.encode(encoding), dtype="u1") - ord("0")


def binary_state_to_state(
    binary_state: Union[np.ndarray, List[Union[int, bool]], str],
) -> np.ndarray:
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
    states = [np.asarray([1 - s, s]) for s in binary_state]
    state = recursive_kron(states)
    return state


def state_to_binary_string(state: Union[int, np.ndarray, sparse.sparray], n: Optional[int] = None) -> str:
    r"""
    Convert a state to a binary state. The binary state is binary string of length :math:`2^n` where :math:`n` is
    the number of particles. The state is a vector of length :math:`2^n` where :math:`n` is the number of particles.

    :param state: State. If the state is an integer, the state is assumed to be a pure state in the computational
        basis and the number of particles must be specified. If the state is a vector, the number of particles is
        inferred from the shape of the vector as :math:`n = \log_2(\text{len}(\text{state}))`.
    :type state: Union[np.ndarray, sparse.sparray]
    :param n: Number of particles. Used only if the state is an integer.
    :type n: Optional[int]
    :return: Binary state as a binary string.
    :rtype: str


    .. Example:
    >>> state_to_binary_string(0, n=2)
    '00'
    >>> state_to_binary_string(1, n=2)
    '01'
    >>> state_to_binary_string(2, n=2)
    '10'
    >>> state_to_binary_string(3, n=2)
    '11'
    >>> state_to_binary_string(np.array([1, 0, 0, 0]))
    '00'
    >>> state_to_binary_string(np.array([0, 1, 0, 0]))
    '01'
    >>> state_to_binary_string(np.array([0, 0, 1, 0]))
    '10'
    >>> state_to_binary_string(np.array([0, 0, 0, 1]))
    '11'
    >>> state_to_binary_string(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    '000'
    """
    if isinstance(state, int):
        assert n is not None, "Number of particles must be specified if the state is an integer."
        assert state < 2**n, f"Invalid state: {state}, must be smaller than 2^n = {2 ** n}."
        return np.binary_repr(state, width=n)
    n_states = np.prod(state.shape)
    n = int(np.log2(n_states))
    assert n_states == 2**n, f"Invalid number of states: {n_states}, must be a power of 2."
    state_number = np.asarray(state.reshape(-1, n_states).argmax(-1)).astype(int).item()
    binary_state = np.binary_repr(state_number, width=n)
    return binary_state


def state_to_binary_state(state: Union[int, np.ndarray, sparse.sparray], n: Optional[int] = None) -> np.ndarray:
    r"""
    Convert a state to a binary state. The binary state is binary string of length :math:`2^n` where :math:`n` is
    the number of particles. The state is a vector of length :math:`2^n` where :math:`n` is the number of particles.

    :param state: State. If the state is an integer, the state is assumed to be a pure state in the computational
        basis and the number of particles must be specified. If the state is a vector, the number of particles is
        inferred from the shape of the vector as :math:`n = \log_2(\text{len}(\text{state}))`.
    :type state: Union[np.ndarray, sparse.sparray]
    :param n: Number of particles. Used only if the state is an integer.
    :type n: Optional[int]
    :return: Binary state as a binary string.
    :rtype: np.ndarray


    .. Example:
    >>> state_to_binary_state(0, n=2)
    array([0, 0])
    >>> state_to_binary_state(1, n=2)
    array([0, 1])
    >>> state_to_binary_state(2, n=2)
    array([1, 0])
    >>> state_to_binary_state(3, n=2)
    array([1, 1])
    >>> state_to_binary_state(np.array([1, 0, 0, 0]))
    array([0, 0])
    >>> state_to_binary_state(np.array([0, 1, 0, 0]))
    array([0, 1])
    >>> state_to_binary_state(np.array([0, 0, 1, 0]))
    array([1, 0])
    >>> state_to_binary_state(np.array([0, 0, 0, 1]))
    array([1, 1])
    >>> state_to_binary_state(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    array([0, 0, 0])
    """
    return binary_string_to_vector(state_to_binary_string(state, n=n))


def binary_string_to_state_number(binary_string: str) -> int:
    r"""
    Convert a binary string to a state number. The binary string is a string of 0s and 1s. The state number is an
    integer.

    :param binary_string: Binary string
    :type binary_string: str
    :return: State number
    :rtype: int
    """
    return int(binary_string, 2)


def get_non_interacting_fermionic_hamiltonian_from_coeffs(
    hamiltonian_coefficients_matrix,
    energy_offset=0.0,
    lib=pnp,
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
    shape = qml.math.shape(hamiltonian_coefficients_matrix)
    ndim = qml.math.ndim(hamiltonian_coefficients_matrix)
    n_particles = int(shape[-2] / 2)
    if ndim == 3:
        hamiltonian = qml.math.stack(
            [energy_offset * backend.eye(2**n_particles, dtype=complex) for _ in range(shape[0])]
        )
    elif ndim == 2:
        hamiltonian = energy_offset * backend.eye(2**n_particles, dtype=complex)
    else:
        raise ValueError(f"hamiltonian_coefficients_matrix must be of dimension 2 or 3. Got {ndim} dimension.")

    for mu in range(2 * n_particles):
        for nu in range(2 * n_particles):
            c_mu = get_majorana(mu, n_particles)
            c_nu = get_majorana(nu, n_particles)
            hamiltonian[..., :, :] += -1j * hamiltonian_coefficients_matrix[..., mu, nu] * (c_mu @ c_nu)
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
        for j in range(i + 1, n):
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
    imag_matrix = qml.math.imag(__matrix)
    zero_like = math.convert_and_cast_like(0.0, imag_matrix)
    return qml.math.allclose(imag_matrix, zero_like, atol=eps)


def decompose_matrix_into_majoranas(
    __matrix: np.ndarray, majorana_getter: Optional[MajoranaGetter] = None
) -> np.ndarray:
    r"""
    Decompose a matrix into Majorana operators. The matrix is decomposed as

    .. math::
        \mathbf{M} = \sum_{i=0}^{2^{n}-1} m_i c_i

    where :math:`\mathbf{M}` is the matrix, :math:`m_i` are the coefficients of the matrix, :math:`n` is the number
    of particles and :math:`c_i` are the Majorana operators.

    :param __matrix: Input matrix
    :type __matrix: np.ndarray
    :param majorana_getter: Majorana getter
    :type majorana_getter: Optional[MajoranaGetter]
    :return: Coefficients of the Majorana operators
    :rtype: np.ndarray
    """
    shape = qml.math.shape(__matrix)
    n_states = shape[-2]
    n = int(np.log2(n_states))
    assert n_states == 2**n, f"Invalid number of states: {n_states}, must be a power of 2."
    assert n == 2, f"Invalid number of particles: {n}, must be 2."
    assert shape[-2:] == (
        n_states,
        n_states,
    ), f"Invalid shape for matrix: {shape}, must be square or batched."
    if majorana_getter is None:
        get_majorana_func = partial(get_majorana, n=n)
    else:
        get_majorana_func = majorana_getter.__getitem__
    majorana_tensor = qml.math.stack([get_majorana_func(i) for i in range(2 * n)])
    return qml.math.trace(__matrix @ majorana_tensor, axis1=-2, axis2=-1) / n_states


def decompose_state_into_majorana_indexes(
    __state: Union[int, np.ndarray, sparse.sparray], n: Optional[int] = None
) -> np.ndarray:
    r"""
    Decompose a state into Majorana operators. The state is decomposed as

    .. math::
        |x> = c_{2p_{1}} ... c_{2p_{\ell}} |0>

    where :math:`|x>` is the state, :math:`c_i` are the Majorana operators, :math:`p_i` are the indices of the
    Majorana operators and :math:`\ell` is the hamming weight of the state.

    Note: The state must be a pure state in the computational basis.

    :param __state: Input state
    :type __state: Union[int, np.ndarray, sparse.sparray]
    :param n: Number of particles. Used only if the state is an integer.
    :type n: Optional[int]
    :return: Indices of the Majorana operators
    :rtype: np.ndarray
    """
    binary_state = state_to_binary_string(__state, n=n)
    return decompose_binary_state_into_majorana_indexes(binary_state)


def decompose_binary_state_into_majorana_indexes(
    __binary_state: Union[np.ndarray, List[Union[int, bool]], str],
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
    action_matrix_t = qml.math.einsum("...ij->...ji", action_matrix)
    transition_matrix = 0.5 * (action_matrix_t[..., ::2, :] + 1j * action_matrix_t[..., 1::2, :])
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
        block_diagonal_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array([[1, 1j], [-1j, 1]])
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
    from ..matchgate_parameter_sets import MatchgateHamiltonianCoefficientsParams

    params = MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
    return np.array(
        [
            [
                -2j * (params.h0 + params.h5),
                0,
                0,
                2 * (params.h4 - params.h1) + 2j * (params.h2 + params.h3),
            ],
            [
                0,
                2j * (params.h0 - params.h5),
                2j * (params.h3 - params.h2) - 2 * (params.h1 + params.h4),
                0,
            ],
            [
                0,
                2 * (params.h1 + params.h4) + 2j * (params.h3 - params.h2),
                2j * (params.h5 - params.h0),
                0,
            ],
            [
                2 * (params.h1 - params.h4) + 2j * (params.h2 + params.h3),
                0,
                0,
                -2j * (params.h0 + params.h5),
            ],
        ],
        dtype=complex,
    )


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
    return qml.math.expm(1j * qml.math.cast(matrix, dtype=complex))


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
    n_states = int(np.log2(qml.math.shape(state)[-1]))
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


def get_all_subclasses(__class, include_base_cls: bool = False) -> set:
    r"""
    Get all the subclasses of a class.

    :param __class: Class
    :type __class: Any
    :param include_base_cls: Include the base class in the set of subclasses
    :type include_base_cls: bool
    :return: Subclasses
    :rtype: set
    """
    subclasses: set = set({})
    for subclass in __class.__subclasses__():
        subclasses.add(subclass)
        subclasses |= get_all_subclasses(subclass)
    if include_base_cls:
        subclasses.add(__class)
    return subclasses


def make_wires_continuous(wires: Union[Wires, np.ndarray]):
    if isinstance(wires, Wires):
        wires_array = wires.toarray()
    else:
        wires_array = np.asarray(wires)
    min_wire, max_wire = np.min(wires_array), np.max(wires_array)
    return Wires(range(min_wire, max_wire + 1))


def make_single_particle_transition_matrix_from_gate(u: Any, majorana_getter: Optional[MajoranaGetter] = None) -> Any:
    r"""
    Compute the single particle transition matrix. This matrix is the matrix :math:`R` such that

    .. math::
        R_{\mu\nu} &= \frac{1}{4} \text{Tr}{\left(U c_\mu U^\dagger\right)c_\nu}

    where :math:`U` is the matchgate and :math:`c_\mu` is the :math:`\mu`-th Majorana operator.

    :Note: This operation is of polynomial complexity only when the number of particles is equal or less than 2
        and of exponential complexity otherwise.

    :param u: Matchgate matrix of shape (..., 2^n, 2^n)
    :param majorana_getter: Majorana getter of n particles
    :type majorana_getter: Optional[MajoranaGetter]
    :return: The single particle transition matrix of shape (..., 2n, 2n)
    """
    if majorana_getter is None:
        majorana_getter = MajoranaGetter(n=int(np.log2(u.shape[-1])))
    # majorana_tensor.shape: (2n, 2^n, 2^n)
    majorana_tensor = qml.math.stack([majorana_getter[i] for i in range(2 * majorana_getter.n)])
    u_c = qml.math.einsum(
        "...ij,mjq->...miq",
        u,
        majorana_tensor,
        optimize="optimal",
    )
    u_c_u_dagger = qml.math.einsum(
        "...miq,...kq->...mik",
        u_c,
        qml.math.conjugate(u),
        optimize="optimal",
    )
    u_c_u_dagger_c = qml.math.einsum("...kij,mjq->...kmiq", u_c_u_dagger, majorana_tensor, optimize="optimal")
    u_c_u_dagger_c_traced = qml.math.einsum("...ii", u_c_u_dagger_c, optimize="optimal")
    return u_c_u_dagger_c_traced / qml.math.shape(majorana_tensor)[-1]


def get_eigvals_on_z_basis(
    op: Operation,
    raise_on_failure: bool = False,
    options_on_failure: Optional[dict] = None,
) -> TensorLike:
    r"""
    Get the eigenvalues of the operator on the Z basis.
    At first, the eigenvalues are computed by extracting the diagonal elements of the operator matrix.
    If the computation fails, the eigenvalues are computed by calling the `qml.eigvals` function.
    The problem with the last one is that we have no guarantee that the eigenvalues are ordered and
    based on the Z basis.

    :param op: Operator
    :type op: qml.Operation
    :param raise_on_failure: Raise an exception if the computation fails. Default is False.
    :type raise_on_failure: bool
    :param options_on_failure: Options to pass to the `qml.eigvals` function if the computation fails.
    :type options_on_failure: Optional[dict]

    :return: Eigenvalues of the operator on the Z basis.
    :rtype: qml.TensorLike
    """
    try:
        op_matrix = op.matrix()
        diag_idx = np.diag_indices(2 ** len(op.wires))
        eigvals_on_z_basis = op_matrix[..., diag_idx[0], diag_idx[1]]
    except Exception as e:
        if raise_on_failure:
            raise e
        options_on_failure = options_on_failure or {}
        eigvals_on_z_basis = qml.eigvals(op, **options_on_failure)
    return eigvals_on_z_basis
