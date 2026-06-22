import itertools

import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_MAP = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_list(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def majoranas(n):
    """JW Majoranas: c_{2k} = Z_{<k} X_k, c_{2k+1} = Z_{<k} Y_k."""
    return [kron_list([Z] * k + [P] + [I2] * (n - k - 1)) for k in range(n) for P in (X, Y)]


def pfaffian(matrix):
    """Recursive complex-capable Pfaffian (small matrices); the test ground truth."""
    m = matrix.shape[0]
    if m == 0:
        return 1.0 + 0j
    if m % 2 == 1:
        return 0.0 + 0j
    if m == 2:
        return matrix[0, 1]
    total = 0.0 + 0j
    rest = list(range(1, m))
    for pos, j in enumerate(rest):
        sub = [k for k in rest if k != j]
        total += (-1) ** pos * matrix[0, j] * pfaffian(matrix[np.ix_(sub, sub)])
    return total


def random_gaussian_unitary(n, cs, rng):
    """Random matchgate (Gaussian) unitary on n qubits."""
    d = 2 * n
    a = rng.normal(size=(d, d))
    a = a - a.T
    H = 0.25 * sum(a[m, u] * cs[m] @ cs[u] for m in range(d) for u in range(m + 1, d))
    w, V = np.linalg.eigh(1j * H)
    return V @ np.diag(np.exp(-1j * w)) @ V.conj().T


def sptm_of(U, cs, n):
    """Single-particle transition matrix Q of a Gaussian unitary U (Lambda -> Q^T Lambda Q)."""
    d = 2 * n
    return np.array([[np.trace(U @ cs[m] @ U.conj().T @ cs[u]) / 2**n for u in range(d)] for m in range(d)]).real


def basis_state(n, bits):
    v = np.zeros(2**n, dtype=complex)
    v[int("".join(map(str, bits)), 2)] = 1.0
    return v


def product_state(n, rng):
    """Random product state (returns the 2**n vector and the (n, 2) per-qubit amplitudes)."""
    amps = []
    psi = np.array([1 + 0j])
    for _ in range(n):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v = v / np.linalg.norm(v)
        amps.append(v)
        psi = np.kron(psi, v)
    return psi, np.stack(amps)


def swap_matrix(n, j, k):
    """Qubit SWAP gate on wires (j, k) as a 2**n x 2**n permutation."""
    P = np.zeros((2**n, 2**n), dtype=complex)
    for bits in itertools.product((0, 1), repeat=n):
        src = int("".join(map(str, bits)), 2)
        nb = list(bits)
        nb[j], nb[k] = nb[k], nb[j]
        P[int("".join(map(str, nb)), 2), src] = 1
    return P


def number_ops(n, cs):
    """Single-mode number operators n_k = (1 + i c_{2k} c_{2k+1}) / 2."""
    return [0.5 * (np.eye(2**n) + 1j * cs[2 * k] @ cs[2 * k + 1]) for k in range(n)]


def fswap_matrix(n, j, k, num):
    """fSWAP = SWAP . CZ on wires (j, k)."""
    cz = np.eye(2**n) - 2 * num[j] @ num[k]
    return swap_matrix(n, j, k) @ cz


def phys_cov_disp(phi, cs):
    """Physical covariance Lambda (2n, 2n) and displacement d (2n,) of a (possibly displaced) state."""
    d = len(cs)
    L = np.zeros((d, d), dtype=complex)
    for mu in range(d):
        for nu in range(d):
            if mu != nu:
                L[mu, nu] = 1j * (phi.conj() @ cs[mu] @ cs[nu] @ phi)
    dv = np.array([phi.conj() @ cs[mu] @ phi for mu in range(d)])
    return L.real, dv.real


def pauli_to_majorana_bruteforce(pstr, n, cs):
    """Return (sorted indices, phase) with P = phase * c_{mu1}...c_{mum} by brute force."""
    op = kron_list([PAULI_MAP[ch] for ch in pstr])
    for m in range(0, 2 * n + 1):
        for sub in itertools.combinations(range(2 * n), m):
            mono = np.eye(2**n, dtype=complex)
            for mu in sub:
                mono = mono @ cs[mu]
            r = np.trace(mono.conj().T @ op) / 2**n
            if abs(abs(r) - 1) < 1e-9:
                return np.array(sub, dtype=int), r
    raise RuntimeError(f"Pauli string {pstr} not matched to a Majorana monomial")
