from functools import cached_property
from typing import Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import StatePrepBase
from ...typing import TensorLike
from pennylane.wires import Wires, WiresLike


class ProductState(StatePrepBase):
    num_wires = None
    grad_method = None
    ATOL = 1e-12

    @staticmethod
    def compute_decomposition(state, wires):
        r"""
        Decompose into one single-qubit ``StatePrep`` per qubit. This lets
        devices that don't natively support :class:`ProductState` execute it
        by falling back to the standard single-qubit state-prep machinery.
        """
        state = qml.math.asarray(state)
        # state has shape (n, 2): row k is the (alpha_k, beta_k) pair for qubit k.
        ops = []
        for k, w in enumerate(wires):
            ops.append(qml.StatePrep(state[k], wires=[w]))
        return ops


    @classmethod
    def from_basis_state(
            cls,
            basis_state,
            wires: Optional[WiresLike] = None,
    ) -> "ProductState":
        r"""
        Construct a :class:`ProductState` from a computational-basis state.

        :param basis_state: A :class:`pennylane.BasisState` operation **or** a
            1-D integer tensor-like of length ``n`` whose :math:`k`-th entry is
            0 or 1 (the bit for qubit *k*).
        :param wires: Wire labels for the resulting operation.  Ignored when
            *basis_state* is a :class:`~pennylane.BasisState` (the wires are
            taken from that object instead).
        :returns: A :class:`ProductState` encoding the same computational-basis
            state.
        """
        if isinstance(basis_state, qml.BasisState):
            bits = qml.math.toarray(basis_state.parameters[0]).astype(int)
            wires = basis_state.wires
        else:
            assert wires is not None, "Must provide wires if basis_state is not a BasisState"
            bits = qml.math.toarray(qml.math.asarray(basis_state)).astype(int)
            wires = Wires(wires)

        n = len(wires)
        if bits.shape != (n,):
            raise ValueError(
                f"basis_state must have shape ({n},) matching wires, got {tuple(bits.shape)}."
            )
        # |0> -> [1, 0],  |1> -> [0, 1]
        state = np.zeros((n, 2), dtype=complex)
        state[bits == 0, 0] = 1.0
        state[bits == 1, 1] = 1.0
        return cls(state, wires=wires, validate_norm=False)


    def __init__(
            self,
            state: TensorLike,
            wires: WiresLike,
            validate_norm: bool = True,
            id: Optional[str] = None,
    ):
        """

        TODO: Add control of dtypes

        :param state:
        :param wires:
        :param validate_norm:
        :param id:
        """
        wires = Wires(wires)
        state = qml.math.asarray(state)

        n = len(wires)
        if state.shape == (2 * n,):
            state = qml.math.reshape(state, (n, 2))
        elif state.shape == (n, 2):
            pass  # already canonical
        else:
            raise ValueError(
                f"ProductState expects either a flat state vector of length "
                f"2 * n_wires = {2 * n} or an (n_wires, 2) = ({n}, 2) matrix, "
                f"got shape {tuple(state.shape)}. Each qubit contributes one "
                f"pair (alpha_k, beta_k) of complex amplitudes."
            )

        # Cast to complex if not already
        if not qml.math.iscomplexobj(state):
            state = qml.math.cast(state, dtype=complex)

        if validate_norm:
            # Per-qubit norms: rows of the (n, 2) matrix
            norms_sq = qml.math.sum(qml.math.abs(state) ** 2, axis=-1)
            if not qml.math.allclose(norms_sq, 1.0, atol=self.ATOL):
                bad = qml.math.where(qml.math.abs(norms_sq - 1.0) > self.ATOL)[0]
                raise ValueError(
                    f"Per-qubit amplitudes must each have unit norm. Qubits "
                    f"{list(np.asarray(bad))} have norms^2 != 1 within atol={self.ATOL}. "
                )

        super().__init__(state, wires=wires, id=id)

    @cached_property
    def covariance_matrix(self) -> TensorLike:
        r"""
        Majorana covariance matrix :math:`\Lambda_0` of the prepared state.

        Returns a real antisymmetric ``(2n, 2n)`` tensor with the convention

        .. math::
            (\Lambda_0)_{\mu\nu} = i\, \langle\Psi|\, c_\mu c_\nu\, |\Psi\rangle,
            \quad \mu \neq \nu,

        using the Jordan-Wigner mapping
        :math:`c_{2k} = Z_{<k} X_k`, :math:`c_{2k+1} = Z_{<k} Y_k` (0-indexed
        Majoranas and qubits).

        For a product state, the matrix has a simple closed form in terms of
        the per-qubit Bloch vectors :math:`(x_k, y_k, z_k)`:

        * Same-qubit entries: :math:`(\Lambda_0)_{2k, 2k+1} = -z_k`.
        * Different-qubit entries (for :math:`j < k`):

          .. math::
              \begin{aligned}
              (\Lambda_0)_{2j,\,2k}     &= +y_j\, p_{jk}\, x_k, \\
              (\Lambda_0)_{2j,\,2k+1}   &= +y_j\, p_{jk}\, y_k, \\
              (\Lambda_0)_{2j+1,\,2k}   &= -x_j\, p_{jk}\, x_k, \\
              (\Lambda_0)_{2j+1,\,2k+1} &= -x_j\, p_{jk}\, y_k,
              \end{aligned}

          where :math:`p_{jk} = \prod_{j < l < k} z_l` is the parity-string
          factor (and :math:`p_{jk} = 1` if :math:`k = j+1`).

        The remaining entries are filled by antisymmetry.
        """
        # data[0] is the canonical (n, 2) matrix of per-qubit amplitudes.
        psi = self.data[0]
        n = len(self.wires)
        alpha = psi[:, 0]
        beta = psi[:, 1]

        # Bloch coordinates per qubit
        x = 2.0 * qml.math.real(qml.math.conj(alpha) * beta)             # <X_k>
        y = 2.0 * qml.math.imag(qml.math.conj(alpha) * beta)             # <Y_k>
        z = qml.math.abs(alpha) ** 2 - qml.math.abs(beta) ** 2           # <Z_k>

        # Convert to a real torch.Tensor on the appropriate dtype.
        x = qml.math.cast(qml.math.real(x), dtype=float)
        y = qml.math.cast(qml.math.real(y), dtype=float)
        z = qml.math.cast(qml.math.real(z), dtype=float)
        x = self._as_torch(x)
        y = self._as_torch(y)
        z = self._as_torch(z)

        cov = torch.zeros((2 * n, 2 * n), dtype=x.dtype, device=x.device)

        # Build the upper-triangular entries; antisymmetrise at the end.
        # Same-qubit entries: (Lambda_0)_{2k, 2k+1} = -z_k.
        for k in range(n):
            cov[2 * k, 2 * k + 1] = -z[k]

        # Different-qubit upper-triangle blocks (j < k).
        # c_{2j}   = Z_{<j} X_j,  c_{2j+1} = Z_{<j} Y_j  (0-indexed).
        # For j<k, after cancelling Z strings on qubits < j:
        #   C_{2j}   C_{2k}   = -i Y_j (x) Z_{j+1..k-1} (x) X_k
        #   C_{2j}   C_{2k+1} = -i Y_j (x) Z_{j+1..k-1} (x) Y_k
        #   C_{2j+1} C_{2k}   = +i X_j (x) Z_{j+1..k-1} (x) X_k
        #   C_{2j+1} C_{2k+1} = +i X_j (x) Z_{j+1..k-1} (x) Y_k
        # Then (Lambda_0)_{mu,nu} = i * <C_mu C_nu>.
        for j in range(n):
            for k in range(j + 1, n):
                if k == j + 1:
                    p = torch.tensor(1.0, dtype=x.dtype, device=x.device)
                else:
                    p = torch.prod(z[j + 1: k])

                cov[2 * j,     2 * k]     = +y[j] * p * x[k]
                cov[2 * j,     2 * k + 1] = +y[j] * p * y[k]
                cov[2 * j + 1, 2 * k]     = -x[j] * p * x[k]
                cov[2 * j + 1, 2 * k + 1] = -x[j] * p * y[k]

        # Antisymmetrise: Lambda <- Lambda - Lambda^T
        cov = cov - qml.math.einsum("...ij->...ji", cov)
        return cov

    def as_basis_state(self) -> qml.BasisState:
        r"""
        Return an equivalent :class:`pennylane.BasisState` operation.

        :raises ValueError: if this state is not a computational-basis state.
        """
        bits = self._basis_state_bits()
        return qml.BasisState(bits, wires=self.wires)

    def state_vector(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        r"""
        Full :math:`2^n`-dim state vector obtained by tensoring per-qubit factors.

        Implements the :class:`pennylane.operation.StatePrepBase` contract so
        non-NIF devices (e.g. ``default.qubit``) can fall back to the explicit
        Kronecker product when the ``ProductState`` op is executed.
        """
        # data[0] is the canonical (n, 2) matrix.
        per_qubit = self.data[0]
        # Kronecker product across qubits
        psi = per_qubit[0]
        for k in range(1, per_qubit.shape[0]):
            psi = qml.math.kron(psi, per_qubit[k])

        if wire_order is None or Wires(wire_order) == self.wires:
            return psi

        n = len(self.wires)
        wire_order = Wires(wire_order)
        if not set(self.wires).issubset(set(wire_order)):
            raise ValueError(
                "wire_order must contain all of this operation's wires."
            )
        # Embed into the larger Hilbert space
        n_total = len(wire_order)
        if n_total == n:
            # Just a permutation among our own wires
            perm = [list(self.wires).index(w) for w in wire_order]
            psi_reshaped = qml.math.reshape(psi, (2,) * n)
            psi_reshaped = qml.math.transpose(psi_reshaped, perm)
            return qml.math.reshape(psi_reshaped, (2 ** n,))
        # If wire_order is strictly larger, pad with |0>s on the extra wires.
        # Build extra |0> factors and Kron in the right positions.
        extra_wires = [w for w in wire_order if w not in self.wires]
        extra = qml.math.asarray([1.0, 0.0], dtype=psi.dtype)
        full = psi
        for _ in extra_wires:
            full = qml.math.kron(full, extra)
        # Now full lives on (self.wires + extra_wires); permute to wire_order
        new_order = list(self.wires) + list(extra_wires)
        perm = [new_order.index(w) for w in wire_order]
        full = qml.math.reshape(full, (2,) * n_total)
        full = qml.math.transpose(full, perm)
        return qml.math.reshape(full, (2 ** n_total,))

    def _as_torch(self, x):
        """
        Convert a tensor-like to a real torch.Tensor, no-op if already one.

        TODO: control dtype
        """
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(np.asarray(x), dtype=torch.float64)

    def _basis_state_bits(self) -> np.ndarray:
        r"""Integer bit array. Raises :class:`ValueError` if not a basis state."""
        if not self.is_basis_state:
            raise ValueError(
                "ProductState is not a computational-basis state. Use "
                "`is_basis_state` to check before calling `as_basis_state()`."
            )
        state = qml.math.toarray(self.data[0])  # shape (n, 2)
        alpha = state[:, 0]
        beta = state[:, 1]
        bits = (np.abs(beta) > np.abs(alpha)).astype(int)
        return bits

    @cached_property
    def is_basis_state(self) -> bool:
        r"""
        Return ``True`` iff every qubit is in a computational-basis state,
        i.e. one of :math:`\alpha_k, \beta_k` is zero within :attr:`atol`.

        This is independent of any global phase on each qubit: a qubit in
        :math:`-|1\rangle` is still considered to be in basis state
        :math:`|1\rangle`.
        """
        state = qml.math.asarray(self.data[0])  # shape (n, 2)
        alpha = state[:, 0]
        beta = state[:, 1]
        alpha_zero = qml.math.abs(alpha) < self.ATOL
        beta_zero = qml.math.abs(beta) < self.ATOL
        return bool(qml.math.all(alpha_zero | beta_zero))

    @property
    def num_params(self) -> int:
        return 1

    @property
    def ndim_params(self) -> tuple:
        return (2,)


