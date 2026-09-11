from typing import List, Optional

import torch

from ...typing import TensorLike

EXACT_DEAD_CUT_TOL = 1e-14


def _validated_weights(weights: TensorLike) -> torch.Tensor:
    """Return ``weights`` as a tensor after checking it can carry branch populations.

    :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
    :return: The weight matrix as a tensor, with the autograd graph left untouched.
    :rtype: torch.Tensor
    :raises ValueError: if ``weights`` is not square in its two leading axes, or if it has no
        entries because ``chi`` is zero or a batch axis has length zero.
    """
    weights_tensor = torch.as_tensor(weights)
    if weights_tensor.ndim < 2 or weights_tensor.shape[0] != weights_tensor.shape[1]:
        raise ValueError(f"expected a weight matrix of shape (chi, chi, ...), got shape {tuple(weights_tensor.shape)}.")
    if weights_tensor.numel() == 0:
        raise ValueError(
            "the weight matrix has no entries, so no branch population is defined: chi is zero or a "
            f"batch axis has length zero. Got shape {tuple(weights_tensor.shape)}."
        )
    return weights_tensor


class RankBudget:
    r"""Policy deciding which branches survive a branching event.

    Both cuts are read off the diagonal of the weight matrix, whose entry
    :math:`W_{\alpha\alpha} = |z_\alpha|^2 \braket{\Phi_\alpha}{\Phi_\alpha}` is the population of
    branch :math:`\alpha`:

    - **Zero-weight branches**, :math:`|W_{\alpha\alpha}| < \varepsilon_{\mathrm{weight}}`, are removed.
      This is what keeps the projector route to ``CZ`` cheap on branches where the gate acts trivially.
      A cut is exact only when the population removed is below :data:`EXACT_DEAD_CUT_TOL`, since
      removing a branch deletes interference of size ``sqrt(population)``.
    - **Rank budget** :math:`\chi_{\mathrm{th}}`: keep the ``chi_th`` branches of largest population and
      remove the rest.

    With ``chi_th = None`` only the exact cut applies. The complementary control is
    :class:`~matchcake.devices.branch_tensor.collinear_merger.CollinearMerger`, which fuses nearly
    parallel branches.

    Batched weights are reduced by the maximum over the batch, so a branch survives as soon as any
    batch element needs it.

    References: Gaussian-rank / stabilizer-rank-style simulators (Dias-Koenig, arXiv:2307.12912;
    Cudby-Strelchuk, arXiv:2307.08551).

    :param weight_tol: Branches with ``|W_aa|`` below this are cut as dead.
    :param chi_th: Maximum number of branches to keep, or ``None`` for no budget (exact).
    """

    @staticmethod
    def branch_populations(weights: TensorLike) -> torch.Tensor:
        """Per-branch population ``|W_aa|``, reduced by the maximum over any batch axes.

        Detached from autograd: pruning is a discrete decision, and comparing a grad-tracking tensor
        would both leak the graph into control flow and warn.

        :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
        :return: Real tensor of shape ``(chi,)``.
        :rtype: torch.Tensor
        :raises ValueError: if ``weights`` is not a non-empty square weight matrix.
        """
        weights_tensor = _validated_weights(weights).detach()
        n_branches = int(weights_tensor.shape[0])
        index = torch.arange(n_branches, device=weights_tensor.device)
        diagonal = torch.abs(weights_tensor[index, index])  # (chi, ...)
        return diagonal.reshape(n_branches, -1).amax(dim=-1)

    def __init__(self, weight_tol: float = 1e-12, chi_th: Optional[int] = None):
        if weight_tol < 0:
            raise ValueError(f"weight_tol must be non-negative, got {weight_tol}.")
        if chi_th is not None and chi_th < 1:
            raise ValueError(f"chi_th must be at least 1 when set, got {chi_th}.")
        self.weight_tol = float(weight_tol)
        self.chi_th = None if chi_th is None else int(chi_th)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weight_tol={self.weight_tol}, chi_th={self.chi_th})"

    def alive_branches(self, weights: TensorLike) -> List[int]:
        """Indices of the branches carrying a nonzero amplitude, in increasing order.

        This is the exact cut alone, without the rank budget. At least one branch is always kept (the
        strongest), so a state never collapses to zero branches through a rounding accident.

        :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
        :return: Sorted list of the surviving branch indices.
        :rtype: List[int]
        """
        populations = self.branch_populations(weights)
        alive = [index for index in range(populations.shape[0]) if float(populations[index]) >= self.weight_tol]
        if not alive:
            alive = [int(torch.argmax(populations))]
        return alive

    def select(self, weights: TensorLike) -> List[int]:
        """Indices of the branches that survive both cuts, in increasing order.

        :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
        :return: Sorted list of surviving branch indices.
        :rtype: List[int]
        """
        alive = self.alive_branches(weights)
        if self.chi_th is None or len(alive) <= self.chi_th:
            return alive
        populations = self.branch_populations(weights)
        ranked = sorted(alive, key=lambda index: float(populations[index]), reverse=True)
        return sorted(ranked[: self.chi_th])
