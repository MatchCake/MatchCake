from typing import Optional

import torch


@torch.no_grad()
def sample_outcomes(
    m_fixed: torch.Tensor,
    shots: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Draw ``shots`` outcome bitstrings from ``p(y) = 2**-k * |Pf(M + Lambda_y)|``.

    Uses the closed-form conditional ``P(y_j = b | prefix) = |m + s_b| / 2`` with
    ``s_b = 2b - 1`` and ``m`` the current Schur complement's ``(0, 1)`` entry, eliminating only
    the sampled branch. Cost is ``O(k^3)`` per shot and no Pfaffian is evaluated, in contrast to
    the chain-rule samplers in :mod:`~matchcake.devices.sampling_strategies`, which pay a
    Pfaffian marginal per candidate prefix.

    The bit convention is ``(Lambda_y)[2j, 2j+1] = 2 y_j - 1``, matching
    :func:`~matchcake.utils.covariance.basis_state_covariance_block` and MatchCake's big-endian
    ``NIFDevice.states_to_binary`` ordering.

    :param m_fixed: Real skew-symmetric covariance ``(..., 2k, 2k)`` whose implied ``p`` is a
        distribution (e.g. a physical ``Q^T Lambda_x Q`` family).
    :param shots: Number of samples (``>= 1``).
    :param generator: Optional ``torch.Generator`` for reproducibility.
    :return: Bit tensor of shape ``(shots, ..., k)``.
    :rtype: torch.Tensor
    :raises ValueError: if ``m_fixed`` is complex, is not a batch of even-sized square matrices,
        or if ``shots < 1``.
    """
    if m_fixed.dtype.is_complex:
        raise ValueError("sampling requires a real covariance family")
    if m_fixed.ndim < 2 or m_fixed.shape[-1] != m_fixed.shape[-2] or m_fixed.shape[-1] % 2 == 1:
        raise ValueError(f"expected (..., 2k, 2k), got shape {tuple(m_fixed.shape)}")
    if shots < 1:
        raise ValueError("shots must be >= 1")

    n_qubits = m_fixed.shape[-1] // 2
    device = m_fixed.device
    states = m_fixed.unsqueeze(0).expand(shots, *m_fixed.shape).contiguous()
    bits = []
    for level in range(n_qubits):
        pivot_entry = states[..., 0, 1]  # (shots, *batch)
        p_one = (pivot_entry.clamp(-1.0, 1.0) + 1.0) / 2.0
        uniform = torch.rand(p_one.shape, dtype=p_one.dtype, device=device, generator=generator)
        bit = uniform < p_one
        sign = torch.where(bit, torch.ones_like(pivot_entry), -torch.ones_like(pivot_entry))
        pivot = pivot_entry + sign
        bits.append(bit)
        if level < n_qubits - 1:
            safe_pivot = torch.where(pivot.abs() > 0, pivot, torch.ones_like(pivot))
            row_block_0 = states[..., 0, 2:]
            row_block_1 = states[..., 1, 2:]
            update = row_block_1.unsqueeze(-1) * row_block_0.unsqueeze(-2)
            update = update - update.transpose(-1, -2)
            states = states[..., 2:, 2:] + update / safe_pivot[..., None, None]
    return torch.stack(bits, dim=-1).long()
