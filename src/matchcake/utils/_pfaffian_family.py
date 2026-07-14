from typing import Optional

import torch
import torch_pfaffian


def outcome_bits(k: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Enumerate outcome bitstrings in the family ordering.

    The row at index ``i`` holds the bits of ``i`` with ``y_0`` (column 0) the most
    significant bit, so ``index = sum_j y_j * 2^(k-1-j)``. This matches MatchCake's
    ``NIFDevice.states_to_binary`` big-endian convention.

    :param k: Number of measured qubits.
    :param device: Optional device for the returned tensor.
    :return: Bit table of shape ``(2**k, k)``.
    :rtype: torch.Tensor
    """
    indices = torch.arange(2**k, device=device)
    shifts = torch.arange(k - 1, -1, -1, device=device)
    return (indices[:, None] >> shifts[None, :]) & 1


def build_lambda_y(bits: torch.Tensor, *, dtype: torch.dtype, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build block-diagonal ``Lambda_y`` matrices from an outcome-bit tensor.

    Sets ``(Lambda_y)[2j, 2j+1] = 2*y_j - 1`` and the skew-symmetric mirror, matching
    MatchCake's ``ProductStateProbabilityStrategy.build_lambda_y``.

    :param bits: Integer bit tensor of shape ``(..., k)``.
    :param dtype: Target dtype of the returned matrices.
    :param device: Optional device; defaults to ``bits.device``.
    :return: Skew-symmetric block-diagonal matrices of shape ``(..., 2k, 2k)``.
    :rtype: torch.Tensor
    """
    k = bits.shape[-1]
    signs = (2 * bits - 1).to(dtype=dtype if not dtype.is_complex else torch.float64)
    lambda_y = torch.zeros(*bits.shape[:-1], 2 * k, 2 * k, dtype=dtype, device=device or bits.device)
    block_indices = torch.arange(k, device=lambda_y.device)
    lambda_y[..., 2 * block_indices, 2 * block_indices + 1] = signs.to(lambda_y.dtype)
    lambda_y[..., 2 * block_indices + 1, 2 * block_indices] = -signs.to(lambda_y.dtype)
    return lambda_y


class OutcomeFamily:
    """Pfaffians of ``M + Lambda_y`` for all ``y`` in ``{0,1}^k`` via a shared-Schur tree.

    The family is ``M_y = M + Lambda_y`` with ``(Lambda_y)[2j, 2j+1] = 2*y_j - 1``. Instead of
    eliminating each of the ``2**k`` matrices independently, the tree eliminates one 2x2 block per
    level and shares the Schur complement across all outcomes with a common prefix; children of a
    node differ only in the pivot ``pivot_entry +- 1``, so the rank-2 update is computed once per
    node. Total work is ``O(2**k)`` versus ``O(2**k * k^3)`` for the batched elimination, and the
    running pivot products at depth ``j`` are the Pfaffians of the leading principal ``2j x 2j``
    submatrices (the prefix marginals). Everything is differentiable batched torch (no host syncs);
    exact-zero pivots mark the whole subtree as ``Pf = 0`` (phase 0, log ``-inf``). Outcome ordering
    matches :func:`outcome_bits` (``y_0`` most significant).

    Preconditions: ``m_fixed`` is assumed skew-symmetric (``m_fixed = -m_fixed^T``); this is not
    validated (the tree runs in hot loops), and a non-skew input yields meaningless output. The
    sweep is UNPIVOTED, so :meth:`all_slog_pfaffians` and :meth:`all_pfaffians` additionally assume
    no outcome has an EXACT intermediate leading-principal Pfaffian of zero while its full Pfaffian
    is nonzero; such an outcome is reported as a spurious ``0``. This is measure-zero for generic
    inputs but reachable with structured/deterministic complex covariances. It never
    affects :meth:`all_probabilities` or
    :func:`sample_outcomes` on a physical covariance family: a zero pivot there is a zero prefix
    marginal, so pruning the subtree is exact.

    :param m_fixed: Skew-symmetric tensor ``(..., 2k, 2k)``, real or complex. Leading dimensions are
        ordinary batch dimensions (e.g. branch pairs ``(chi, chi)``).
    :param prune_threshold: Pivots with ``abs(pivot) <= prune_threshold`` mark their subtree
        ``Pf = 0``. Default ``0.0`` prunes exact zeros only.
    :param work_dtype: Optional dtype override for the internal sweep (e.g. promote float32 to
        float64 for accuracy).
    """

    def __init__(
        self,
        m_fixed: torch.Tensor,
        *,
        prune_threshold: float = 0.0,
        work_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if m_fixed.ndim < 2 or m_fixed.shape[-1] != m_fixed.shape[-2]:
            raise ValueError(f"expected (..., 2k, 2k), got shape {tuple(m_fixed.shape)}")
        if m_fixed.shape[-1] % 2 == 1:
            raise ValueError("outcome families need an even matrix size 2k")
        if prune_threshold < 0.0:
            raise ValueError("prune_threshold must be >= 0")
        self.m_fixed = m_fixed.to(work_dtype) if work_dtype is not None else m_fixed
        self.k = m_fixed.shape[-1] // 2
        self.prune_threshold = float(prune_threshold)

    def all_slog_pfaffians(self, *, zero_pivot_fallback: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phase, log|Pf|)`` of shape ``(..., 2**k)`` for every outcome.

        Differentiable in ``log|Pf|`` through plain tensor ops; dead outcomes give ``(0, -inf)``.

        :param zero_pivot_fallback: When True, every outcome pruned by an exact-zero pivot is
            recomputed with a pivoted Pfaffian (``torch_pfaffian.slog_pfaffian``), which repairs the
            unpivoted sweep's spurious zeros on general (e.g. complex) matrices. Exact zeros are rare,
            so only those lanes are recomputed. Leave False for physical covariance families, where a
            zero pivot is a genuine zero marginal and the prune is exact and cheap.
        :type zero_pivot_fallback: bool
        :return: A pair ``(phase, log_abs)`` each of shape ``(..., 2**k)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        matrix = self.m_fixed
        dtype = matrix.dtype
        device = matrix.device
        real_dtype = matrix.real.dtype if dtype.is_complex else dtype
        batch_shape = matrix.shape[:-2]
        k = self.k

        states = matrix.reshape(1, *batch_shape, 2 * k, 2 * k)  # node axis is axis 0
        phase = torch.ones((1, *batch_shape), dtype=dtype, device=device)
        log_abs = torch.zeros((1, *batch_shape), dtype=real_dtype, device=device)
        alive = torch.ones((1, *batch_shape), dtype=torch.bool, device=device)
        signs = torch.tensor([-1.0, 1.0], dtype=real_dtype, device=device)
        signs = signs.reshape(1, 2, *([1] * len(batch_shape)))

        for level in range(k):
            n_nodes = states.shape[0]
            pivot_entry = states[..., 0, 1]  # (n, *batch)
            pivot = pivot_entry.unsqueeze(1) + signs.to(dtype)  # (n, 2, *batch); bit y has sign 2y-1
            abs_pivot = pivot.abs()
            child_alive = alive.unsqueeze(1) & (abs_pivot > self.prune_threshold)
            safe_pivot = torch.where(child_alive, pivot, torch.ones_like(pivot))
            safe_abs = torch.where(child_alive, abs_pivot, torch.ones_like(abs_pivot))
            phase = (phase.unsqueeze(1) * (safe_pivot / safe_abs.to(dtype))).reshape(2 * n_nodes, *batch_shape)
            log_abs = (log_abs.unsqueeze(1) + safe_abs.log()).reshape(2 * n_nodes, *batch_shape)
            alive = child_alive.reshape(2 * n_nodes, *batch_shape)

            if level < k - 1:
                row_block_0 = states[..., 0, 2:]
                row_block_1 = states[..., 1, 2:]
                trailing = states[..., 2:, 2:]
                update = row_block_1.unsqueeze(-1) * row_block_0.unsqueeze(-2)
                update = update - update.transpose(-1, -2)
                states = trailing.unsqueeze(1) + update.unsqueeze(1) / safe_pivot.reshape(
                    n_nodes, 2, *batch_shape, 1, 1
                )
                states = states.reshape(2 * n_nodes, *batch_shape, 2 * (k - level - 1), 2 * (k - level - 1))

        phase = torch.where(alive, phase, torch.zeros_like(phase))
        log_abs = torch.where(alive, log_abs, torch.full_like(log_abs, -torch.inf))
        phase = phase.movedim(0, -1)  # outcome index goes last: (..., 2^k)
        log_abs = log_abs.movedim(0, -1)
        if zero_pivot_fallback:
            phase, log_abs = self._recompute_zero_pivot_lanes(phase, log_abs)
        return phase, log_abs

    def all_pfaffians(self, *, zero_pivot_fallback: bool = False) -> torch.Tensor:
        """Return linear-domain Pfaffians ``(..., 2**k)``; differentiable (``phase * exp(log)``).

        :param zero_pivot_fallback: See :meth:`all_slog_pfaffians`.
        :type zero_pivot_fallback: bool
        :return: Pfaffian tensor of shape ``(..., 2**k)``.
        :rtype: torch.Tensor
        """
        phase, log_abs = self.all_slog_pfaffians(zero_pivot_fallback=zero_pivot_fallback)
        return phase * torch.exp(log_abs).to(phase.dtype)

    def all_probabilities(self) -> torch.Tensor:
        """Return ``p(y) = 2**-k * |Pf(M + Lambda_y)|`` of shape ``(..., 2**k)`` (real families).

        :return: Probability tensor of shape ``(..., 2**k)``.
        :rtype: torch.Tensor
        """
        if self.m_fixed.dtype.is_complex:
            raise ValueError("probabilities are defined for real covariance families; use all_slog_pfaffians")
        _, log_abs = self.all_slog_pfaffians()
        log2 = torch.log(torch.tensor(2.0, dtype=log_abs.dtype, device=log_abs.device))
        return torch.exp(log_abs - self.k * log2)

    def _recompute_zero_pivot_lanes(
        self, phase: torch.Tensor, log_abs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute exact-zero-pruned outcome lanes with a pivoted Pfaffian.

        The unpivoted sweep prunes an outcome whenever an intermediate leading-principal Pfaffian is
        exactly zero, even when the full Pfaffian is nonzero. Those lanes have ``log_abs = -inf``; each
        is recomputed from the materialized ``M + Lambda_y`` via ``torch_pfaffian.slog_pfaffian``
        (partial pivoting), which is correct at exact-zero pivots. Only the (rare) pruned lanes are
        materialized, so the amortized cost of the surviving lanes is untouched.

        :param phase: Phase tensor ``(..., 2**k)`` from the sweep.
        :type phase: torch.Tensor
        :param log_abs: Log-magnitude tensor ``(..., 2**k)`` from the sweep.
        :type log_abs: torch.Tensor
        :return: The repaired ``(phase, log_abs)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        dead = torch.isneginf(log_abs)  # exact-zero-pruned lanes
        if not bool(dead.any()):
            return phase, log_abs
        matrix = self.m_fixed
        k = self.k
        n_outcomes = 2**k
        batch_shape = matrix.shape[:-2]
        flat_matrix = matrix.reshape(-1, 2 * k, 2 * k)  # (B, 2k, 2k)
        positions = dead.reshape(-1, n_outcomes).nonzero(as_tuple=False)  # (n_dead, 2): (batch, outcome)
        batch_index, outcome_index = positions[:, 0], positions[:, 1]
        bits = outcome_bits(k, device=matrix.device)[outcome_index]  # (n_dead, k)
        lambda_dead = build_lambda_y(bits, dtype=matrix.dtype, device=matrix.device)  # (n_dead, 2k, 2k)
        recomputed = flat_matrix[batch_index] + lambda_dead
        recomputed_phase, recomputed_log = torch_pfaffian.slog_pfaffian(recomputed)  # pivoted
        flat_phase = phase.reshape(-1, n_outcomes).index_put(
            (batch_index, outcome_index), recomputed_phase.to(phase.dtype)
        )
        flat_log = log_abs.reshape(-1, n_outcomes).index_put(
            (batch_index, outcome_index), recomputed_log.to(log_abs.dtype)
        )
        return flat_phase.reshape(*batch_shape, n_outcomes), flat_log.reshape(*batch_shape, n_outcomes)

    @property
    def n_outcomes(self) -> int:
        """Number of outcomes ``2**k``.

        :return: The outcome count.
        :rtype: int
        """
        return 2**self.k


@torch.no_grad()
def sample_outcomes(
    m_fixed: torch.Tensor,
    shots: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw ``shots`` outcome bitstrings from ``p(y) = 2**-k * |Pf(M + Lambda_y)|``.

    Uses the closed-form conditional ``P(y_j = b | prefix) = |m + s_b| / 2`` with
    ``s_b = 2b - 1`` and ``m`` the current Schur complement's ``(0, 1)`` entry, eliminating only the
    sampled branch (``O(k^3)`` per shot, no full-matrix Pfaffian). Bit convention matches
    :func:`outcome_bits`.

    :param m_fixed: Real skew-symmetric covariance ``(..., 2k, 2k)`` whose implied ``p`` is a
        distribution (e.g. a physical ``Q^T Lambda_x Q`` family).
    :param shots: Number of samples (``>= 1``).
    :param generator: Optional ``torch.Generator`` for reproducibility.
    :return: Bit tensor of shape ``(shots, ..., k)``.
    :rtype: torch.Tensor
    """
    if m_fixed.dtype.is_complex:
        raise ValueError("sampling requires a real covariance family")
    if m_fixed.ndim < 2 or m_fixed.shape[-1] != m_fixed.shape[-2] or m_fixed.shape[-1] % 2 == 1:
        raise ValueError(f"expected (..., 2k, 2k), got shape {tuple(m_fixed.shape)}")
    if shots < 1:
        raise ValueError("shots must be >= 1")

    k = m_fixed.shape[-1] // 2
    device = m_fixed.device
    states = m_fixed.unsqueeze(0).expand(shots, *m_fixed.shape).contiguous()
    bits = []
    for level in range(k):
        pivot_entry = states[..., 0, 1]  # (shots, *batch)
        p_one = (pivot_entry.clamp(-1.0, 1.0) + 1.0) / 2.0
        uniform = torch.rand(p_one.shape, dtype=p_one.dtype, device=device, generator=generator)
        bit = uniform < p_one
        sign = torch.where(bit, torch.ones_like(pivot_entry), -torch.ones_like(pivot_entry))
        pivot = pivot_entry + sign
        bits.append(bit)
        if level < k - 1:
            safe_pivot = torch.where(pivot.abs() > 0, pivot, torch.ones_like(pivot))
            row_block_0 = states[..., 0, 2:]
            row_block_1 = states[..., 1, 2:]
            update = row_block_1.unsqueeze(-1) * row_block_0.unsqueeze(-2)
            update = update - update.transpose(-1, -2)
            states = states[..., 2:, 2:] + update / safe_pivot[..., None, None]
    return torch.stack(bits, dim=-1).long()
