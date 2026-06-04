from functools import cached_property
from typing import Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import StatePrepBase
from pennylane.wires import Wires, WiresLike

from ...typing import TensorLike


class ProductState(StatePrepBase):
    r"""
    Prepare a product state from per-qubit amplitudes.

    A single state is given either as a flat ``(2n,)`` vector or as an
    ``(n, 2)`` matrix of per-qubit ``(alpha_k, beta_k)`` pairs.  A *batch* of
    ``B`` states can also be passed as a ``(B, 2n)`` or ``(B, n, 2)`` array; in
    that case :attr:`batch_size` is ``B`` and the per-state outputs
    (:meth:`state_vector`, :attr:`covariance_matrix`, ...) gain a leading batch
    axis.
    """

    num_wires = None
    grad_method = None
    ATOL = 1e-12

    @staticmethod
    def compute_decomposition(state, wires):
        r"""
        Decompose into a single ``StatePrep`` from the full state vector. This lets
        devices that don't natively support :class:`ProductState` execute it
        by falling back to the standard state-prep machinery.
        """
        op = ProductState(state, wires=wires, validate_norm=False)
        sv = op.state_vector()
        if op.batch_size is None:
            flat = qml.math.reshape(sv, (-1,))
        else:
            flat = qml.math.reshape(sv, (op.batch_size, -1))
        return [qml.StatePrep(flat, wires=wires)]

    @classmethod
    def from_basis_state(
        cls,
        basis_state,
        wires: Optional[WiresLike] = None,
    ) -> "ProductState":
        r"""
        Construct a :class:`ProductState` from a computational-basis state.

        :param basis_state: A :class:`pennylane.BasisState` operation **or** an
            integer tensor-like of shape ``(n,)`` (a single state) or
            ``(batch, n)`` (a batch of states) whose :math:`k`-th entry is
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
        if bits.ndim not in (1, 2) or bits.shape[-1] != n:
            raise ValueError(
                f"basis_state must have shape ({n},) or (batch, {n}) matching wires, got {tuple(bits.shape)}."
            )
        # |0> -> [1, 0],  |1> -> [0, 1].  One-hot encode along a trailing axis.
        state = np.eye(2, dtype=complex)[bits]  # (..., n, 2)
        return cls(state, wires=wires, validate_norm=False)

    def __init__(
        self,
        state: TensorLike,
        wires: WiresLike,
        validate_norm: bool = True,
    ):
        """

        TODO: Add control of dtypes

        :param state: Per-qubit amplitudes as a flat ``(2n,)`` vector, an
            ``(n, 2)`` matrix, or their batched counterparts ``(B, 2n)`` /
            ``(B, n, 2)``.
        :param wires:
        :param validate_norm:
        """
        wires = Wires(wires)
        state = qml.math.asarray(state)

        n = len(wires)
        shape = tuple(state.shape)
        if shape == (2 * n,):
            state = qml.math.reshape(state, (n, 2))
        elif shape == (n, 2):
            pass  # already canonical (single, unbatched)
        elif len(shape) == 2 and shape[1] == 2 * n:
            # Batched flat vectors: (B, 2n) -> (B, n, 2)
            state = qml.math.reshape(state, (shape[0], n, 2))
        elif len(shape) == 3 and shape[1:] == (n, 2):
            pass  # already canonical (batched)
        else:
            raise ValueError(
                f"ProductState expects a flat state vector of length "
                f"2 * n_wires = {2 * n}, an (n_wires, 2) = ({n}, 2) matrix, or "
                f"their batched forms (B, {2 * n}) / (B, {n}, 2), got shape "
                f"{shape}. Each qubit contributes one pair (alpha_k, beta_k) of "
                f"complex amplitudes."
            )

        # Cast to complex if not already (qml.math.iscomplexobj has no torch dispatch)
        is_complex = torch.is_complex(state) if isinstance(state, torch.Tensor) else qml.math.iscomplexobj(state)
        if not is_complex:
            state = qml.math.cast(state, dtype=complex)

        if validate_norm:
            # Per-qubit norms: sum over the trailing (alpha, beta) axis.
            norms_sq = qml.math.sum(qml.math.abs(state) ** 2, axis=-1)
            if not qml.math.allclose(norms_sq, 1.0, atol=self.ATOL):
                bad = np.argwhere(np.asarray(qml.math.abs(norms_sq - 1.0) > self.ATOL))
                raise ValueError(
                    f"Per-qubit amplitudes must each have unit norm. Entries "
                    f"{bad.tolist()} have norms^2 != 1 within atol={self.ATOL}."
                )

        super().__init__(state, wires=wires)

    @cached_property
    def covariance_matrix(self) -> TensorLike:
        r"""
        Majorana covariance matrix :math:`\Lambda_0` of the prepared state.

        Returns a real antisymmetric ``(2n, 2n)`` tensor (or ``(B, 2n, 2n)`` for
        a batched state) with the convention

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
        # data[0] is the canonical (n, 2) matrix of per-qubit amplitudes, or
        # (B, n, 2) when batched.  Selecting the trailing axis keeps any
        # leading batch dimensions intact.
        psi = self.data[0]
        n = len(self.wires)
        alpha = psi[..., 0]  # (..., n)
        beta = psi[..., 1]  # (..., n)

        # Bloch coordinates per qubit
        x = 2.0 * qml.math.real(qml.math.conj(alpha) * beta)  # <X_k>
        y = 2.0 * qml.math.imag(qml.math.conj(alpha) * beta)  # <Y_k>
        z = qml.math.abs(alpha) ** 2 - qml.math.abs(beta) ** 2  # <Z_k>

        # Convert to real torch.Tensors on the appropriate dtype.
        x = qml.math.cast(qml.math.real(x), dtype=float)
        y = qml.math.cast(qml.math.real(y), dtype=float)
        z = qml.math.cast(qml.math.real(z), dtype=float)
        x = self._as_torch(x)
        y = self._as_torch(y)
        z = self._as_torch(z)

        batch_shape = tuple(x.shape[:-1])
        cov = torch.zeros(batch_shape + (2 * n, 2 * n), dtype=x.dtype, device=x.device)

        # Build the upper-triangular entries; antisymmetrise at the end.
        # Same-qubit entries: (Lambda_0)_{2k, 2k+1} = -z_k.
        diag = torch.arange(n, device=x.device)
        cov[..., 2 * diag, 2 * diag + 1] = -z

        # Parity-string factors p_{jk} = prod_{j < l < k} z_l for every pair
        # j < k.  Built out-of-place to avoid autograd version-counter errors:
        # each column is derived directly from z via a suffix cumprod, never by
        # reading a previously in-place-written column of p_mat.
        p_mat_cols = []
        for k in range(n):
            if k <= 1:
                p_mat_cols.append(torch.ones(batch_shape + (n,), dtype=x.dtype, device=x.device))
            else:
                # col[j] = prod(z[j+1 : k]) for j < k-1, else 1.
                # This equals the suffix product of z[1:k] at position j.
                z_slice = z[..., 1:k]  # (..., k-1)
                suffix = torch.flip(torch.cumprod(torch.flip(z_slice, dims=[-1]), dim=-1), dims=[-1])
                ones_pad = torch.ones(batch_shape + (n - k + 1,), dtype=x.dtype, device=x.device)
                p_mat_cols.append(torch.cat([suffix, ones_pad], dim=-1))
        p_mat = torch.stack(p_mat_cols, dim=-1)  # (..., n, n)

        # Different-qubit upper-triangle blocks (j < k), filled in one shot over
        # all pairs given by the upper-triangular indices.
        # c_{2j}   = Z_{<j} X_j,  c_{2j+1} = Z_{<j} Y_j  (0-indexed).
        # For j<k, after cancelling Z strings on qubits < j:
        #   C_{2j}   C_{2k}   = -i Y_j (x) Z_{j+1..k-1} (x) X_k
        #   C_{2j}   C_{2k+1} = -i Y_j (x) Z_{j+1..k-1} (x) Y_k
        #   C_{2j+1} C_{2k}   = +i X_j (x) Z_{j+1..k-1} (x) X_k
        #   C_{2j+1} C_{2k+1} = +i X_j (x) Z_{j+1..k-1} (x) Y_k
        # Then (Lambda_0)_{mu,nu} = i * <C_mu C_nu>.
        rows, cols = np.triu_indices(n, k=1)
        row_idx = torch.as_tensor(rows, device=x.device)
        col_idx = torch.as_tensor(cols, device=x.device)
        p = p_mat[..., row_idx, col_idx]  # (..., n * (n - 1) / 2)

        cov[..., 2 * row_idx, 2 * col_idx] = +y[..., row_idx] * p * x[..., col_idx]
        cov[..., 2 * row_idx, 2 * col_idx + 1] = +y[..., row_idx] * p * y[..., col_idx]
        cov[..., 2 * row_idx + 1, 2 * col_idx] = -x[..., row_idx] * p * x[..., col_idx]
        cov[..., 2 * row_idx + 1, 2 * col_idx + 1] = -x[..., row_idx] * p * y[..., col_idx]

        # Antisymmetrise: Lambda <- Lambda - Lambda^T
        cov = cov - qml.math.einsum("...ij->...ji", cov)
        return cov

    def as_basis_state(self) -> qml.BasisState:
        r"""
        Return an equivalent :class:`pennylane.BasisState` operation.

        :raises ValueError: if this state is not a computational-basis state,
            or if it is batched (``qml.BasisState`` has no batch dimension; use
            :meth:`basis_state_bits` to obtain the per-state bit arrays).
        """
        if self.batch_size is not None:
            raise ValueError(
                "as_basis_state() is not defined for a batched ProductState because "
                "qml.BasisState has no batch dimension. Use `basis_state_bits()` to "
                "obtain the (batch, n) array of bits instead."
            )
        bits = self.basis_state_bits()
        return qml.BasisState(bits, wires=self.wires)

    def state_vector(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        r"""
        Full :math:`2^n`-dim state vector obtained by tensoring per-qubit factors.

        Implements the :class:`pennylane.operation.StatePrepBase` contract so
        non-NIF devices (e.g. ``default.qubit``) can fall back to the explicit
        Kronecker product when the ``ProductState`` op is executed.  For a
        batched state the result carries a leading batch axis.
        """
        # data[0] is the canonical (n, 2) matrix, or (B, n, 2) when batched.
        per_qubit = self.data[0]
        n = len(self.wires)
        batch_shape = tuple(per_qubit.shape[:-2])
        nb = len(batch_shape)

        # Batched Kronecker product across the qubit axis (second-to-last).
        psi_flat = per_qubit[..., 0, :]  # (..., 2)
        for k in range(1, n):
            factor = per_qubit[..., k, :]  # (..., 2)
            psi_flat = qml.math.reshape(
                psi_flat[..., :, None] * factor[..., None, :],
                batch_shape + (-1,),
            )
        psi = qml.math.reshape(psi_flat, batch_shape + (2,) * n)

        if wire_order is None or Wires(wire_order) == self.wires:
            return psi

        wire_order = Wires(wire_order)
        if not set(self.wires).issubset(set(wire_order)):
            raise ValueError("wire_order must contain all of this operation's wires.")
        n_total = len(wire_order)
        if n_total == n:
            # Pure permutation of our wires (batch axes stay leading).
            perm = [list(self.wires).index(w) for w in wire_order]
            full_perm = list(range(nb)) + [nb + p for p in perm]
            return qml.math.transpose(psi, full_perm)
        # Embed into a larger Hilbert space by padding extra wires with |0>.
        extra_wires = [w for w in wire_order if w not in self.wires]
        zero = qml.math.asarray([1.0, 0.0], dtype=psi_flat.dtype)
        full = psi_flat
        for _ in extra_wires:
            full = qml.math.reshape(
                full[..., :, None] * zero,
                batch_shape + (-1,),
            )
        new_order = list(self.wires) + list(extra_wires)
        perm = [new_order.index(w) for w in wire_order]
        full_perm = list(range(nb)) + [nb + p for p in perm]
        return qml.math.transpose(qml.math.reshape(full, batch_shape + (2,) * n_total), full_perm)

    def _as_torch(self, x):
        """
        Convert a tensor-like to a real torch.Tensor, no-op if already one.

        TODO: control dtype
        """
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(np.asarray(x), dtype=torch.float64)

    def basis_state_bits(self) -> np.ndarray:
        r"""Integer bit array. Raises :class:`ValueError` if not a basis state.

        Shape is ``(n,)`` for a single state and ``(batch, n)`` when batched.
        """
        is_basis = self.is_basis_state
        all_basis = is_basis if self.batch_size is None else bool(qml.math.all(is_basis))
        if not all_basis:
            raise ValueError(
                "ProductState is not a computational-basis state. Use "
                "`is_basis_state` to check before calling `as_basis_state()`."
            )
        state = qml.math.toarray(self.data[0])  # (..., n, 2)
        alpha = state[..., 0]
        beta = state[..., 1]
        bits = (np.abs(beta) > np.abs(alpha)).astype(int)  # (..., n)
        return bits

    @cached_property
    def is_basis_state(self):
        r"""
        Whether every qubit is in a computational-basis state, i.e. one of
        :math:`\alpha_k, \beta_k` is zero within :attr:`ATOL`.

        Returns a single ``bool`` for an unbatched state, or a boolean tensor of
        shape ``(batch,)`` when batched (one entry per state).

        This is independent of any global phase on each qubit: a qubit in
        :math:`-|1\rangle` is still considered to be in basis state
        :math:`|1\rangle`.
        """
        state = qml.math.asarray(self.data[0])  # (..., n, 2)
        alpha = state[..., 0]
        beta = state[..., 1]
        alpha_zero = qml.math.abs(alpha) < self.ATOL
        beta_zero = qml.math.abs(beta) < self.ATOL
        # Reduce over the qubit axis only, keeping any batch axis.
        result = qml.math.all(alpha_zero | beta_zero, axis=-1)
        if self.batch_size is None:
            return bool(result)
        return result

    @property
    def num_params(self) -> int:
        return 1

    @property
    def ndim_params(self) -> tuple:
        return (2,)
