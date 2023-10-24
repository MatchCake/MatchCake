import numpy as np


PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULI_I = np.eye(2)


def recursive_kron(__inputs, lib=np):
    if isinstance(lib, str):
        lib = eval(lib)

    if len(__inputs) == 1:
        return __inputs[0]
    elif len(__inputs) == 2:
        return lib.kron(__inputs[0], __inputs[1])
    elif len(__inputs) > 2:
        return lib.kron(__inputs[0], recursive_kron(__inputs[1:]))
    else:
        raise ValueError("Invalid shape for input array")


def get_majorana_mu(k: int, n: int) -> np.ndarray:
    r"""

    Get the Majorana matrix :math:`c_\mu` defined as

    .. math::
        c_{2k-1} = Z^{\otimes k-1} \otimes X \otimes I^{\otimes n-k}

    where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix, :math:`X` is the Pauli X matrix,
    :math:`\otimes` is the Kronecker product, :math:`k` is the index of the Majorana operator and :math:`n` is the
    number of particles.

    :param k: Index of the Majorana operator
    :type k: int
    :param n: Number of particles
    :type n: int
    :return: Majorana matrix :math:`c_\mu`
    :rtype: np.ndarray
    """
    assert k > 0
    assert n > 0
    assert k <= n

    if k == 1:
        return PAULI_X
    elif k == n:
        return recursive_kron([PAULI_Z] * (n - 1) + [PAULI_X])
    else:
        return recursive_kron([PAULI_Z] * (k - 1) + [PAULI_X] + [PAULI_I] * (n - k))


def get_majorana_nu(k: int, n: int) -> np.ndarray:
    r"""

    Get the Majorana matrix :math:`c_\nu` defined as

    .. math::
        c_{2k} = Z^{\otimes k-1} \otimes Y \otimes I^{\otimes n-k}

    where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix, :math:`Y` is the Pauli Y matrix,
    :math:`\otimes` is the Kronecker product, :math:`k` is the index of the Majorana operator and :math:`n` is the
    number of particles.

    :param k: Index of the Majorana operator
    :type k: int
    :param n: Number of particles
    :type n: int
    :return: Majorana matrix :math:`c_\nu`
    :rtype: np.ndarray
    """
    assert k > 0
    assert n > 0
    assert k <= n

    if k == 1:
        return PAULI_Y
    elif k == n:
        return recursive_kron([PAULI_Z] * (n - 1) + [PAULI_Y])
    else:
        return recursive_kron([PAULI_Z] * (k - 1) + [PAULI_Y] + [PAULI_I] * (n - k))


def get_non_interacting_fermionic_hamiltonian_from_coeffs(hamiltonian_coefficients, lib=np):
    r"""

    Compute the non-interacting fermionic Hamiltonian from the coefficients of the Majorana operators.

    .. math::
        H = i \sum_{\mu\neq\nu}^{2n} h_{\mu \nu} c_\mu c_\nu

    where :math:`h_{\mu \nu}` are the coefficients of the Majorana operators :math:`c_\mu` and :math:`c_\nu`,
    :math:`n` is the number of particles, :math:`\mu` and :math:`\nu` are the indices of the Majorana operators, and
    :math:`i` is the imaginary unit.

    :param hamiltonian_coefficients: Coefficients of the Majorana operators
    :param lib: Library to use for the operations
    :return: Non-interacting fermionic Hamiltonian
    """
    if isinstance(lib, str):
        lib = eval(lib)

    hamiltonian = np.zeros((len(hamiltonian_coefficients), len(hamiltonian_coefficients)), dtype=complex)
    n_particles = int(len(hamiltonian_coefficients) / 2)
    for mu in range(len(hamiltonian_coefficients)):
        for nu in range(len(hamiltonian_coefficients)):
            c_mu = get_majorana_mu(mu + 1, n_particles)
            c_nu = get_majorana_nu(nu + 1, n_particles)
            hamiltonian += hamiltonian_coefficients[mu, nu] * c_mu @ c_nu

    return hamiltonian

