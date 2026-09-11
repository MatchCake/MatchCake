from typing import List, Sequence, Tuple

import torch

from ...typing import TensorLike

EXACT_COLLINEAR_TOL = 1e-10


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


def _validated_groups(groups: Sequence[Sequence[int]], n_branches: int) -> List[List[int]]:
    """Return ``groups`` as lists of ints after checking they partition ``range(n_branches)``.

    Nothing downstream can detect a malformed partition: a branch named twice has its weight counted
    twice and a branch named nowhere is silently dropped, both of which change the norm of the state
    without raising.

    :param groups: Candidate partition of ``range(n_branches)``.
    :param n_branches: Number of branches the partition must cover.
    :return: The groups, with every index cast to ``int``.
    :rtype: List[List[int]]
    :raises ValueError: if a group is empty, or if the groups are not a partition of
        ``range(n_branches)``.
    """
    partition = [[int(member) for member in group] for group in groups]
    if any(len(group) == 0 for group in partition):
        raise ValueError(f"groups must not contain an empty group, got {partition}.")
    members = sorted(member for group in partition for member in group)
    if members != list(range(n_branches)):
        raise ValueError(
            f"groups must partition the {n_branches} branches exactly once each, got members "
            f"{members}. A repeated branch is counted twice and a missing one is dropped, either of "
            f"which changes the norm of the state."
        )
    return partition


class CollinearMerger:
    r"""Fuses branches that are parallel in Hilbert space, exact at its default tolerance.

    A branching by a unitary monomial term produces no zero-weight branch, so the exact cut of
    :class:`~matchcake.devices.branch_tensor.rank_budget.RankBudget` has nothing to remove. What it
    produces instead are **collinear** branches, detected directly from the weight matrix by equality
    in the Cauchy-Schwarz inequality, with no covariance comparison:

    .. math::
        \text{collinear}(\alpha, \beta) \iff
        \frac{|W_{\alpha\beta}|}{\sqrt{W_{\alpha\alpha} W_{\beta\beta}}} \ge 1 - \varepsilon_{\mathrm{col}}
        \iff |\braket{\Phi_\alpha}{\Phi_\beta}| = 1 .

    Merging is then just **summing the two rows and the two columns of** :math:`W`. Writing
    :math:`\ket{\Phi_\beta} = e^{i\varphi} \ket{\Phi_\alpha}` and
    :math:`W_{\alpha\beta} = \bar z_\alpha z_\beta \braket{\Phi_\alpha}{\Phi_\beta}`, the fused branch
    carries amplitude :math:`z_\alpha + e^{i\varphi} z_\beta`, whence
    :math:`W'_{\alpha\gamma} = W_{\alpha\gamma} + W_{\beta\gamma}` and
    :math:`W'_{\alpha\alpha} = W_{\alpha\alpha} + W_{\beta\beta} + 2\,\mathrm{Re}\,W_{\alpha\beta}`. The
    relative phase :math:`\arg W_{\alpha\beta}` never has to be extracted: it is already inside
    :math:`W`. The covariance is phase-blind, so the representative's covariance is kept unchanged.

    At its default ``collinear_th`` the merge is **exact**; raising it turns it into a second,
    independent approximation knob that lowers the recorded :math:`\chi`, complementary in kind to
    :class:`~matchcake.devices.branch_tensor.rank_budget.RankBudget` (which cuts the twigs rather than
    fusing parallel ones). Merging does not subsume the projector: two genuinely independent branches
    can still sum to a single Gaussian state.

    Batched weights are reduced by the minimum over the batch, so branches merge only when they are
    collinear for every batch element.

    References: Gaussian-rank simulators (Dias-Koenig, arXiv:2307.12912; Cudby-Strelchuk,
    arXiv:2307.08551); NOCI-style branch reweighting (Thom-Head-Gordon, JCP 2009).

    :param collinear_th: The tolerance :math:`\varepsilon_{\mathrm{col}}` on the normalized Gram.
        Defaults to :data:`EXACT_COLLINEAR_TOL`, where the merge is exact.
    """

    @staticmethod
    def apply_groups(
        covariances: TensorLike,
        weights: TensorLike,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[TensorLike, TensorLike]:
        """Fuse each group into its first member by summing the group's rows and columns of ``W``.

        :param covariances: Branch covariance tensor of shape ``(chi, ..., D, D)``.
        :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
        :param groups: Partition of ``range(chi)`` into groups of mutually collinear branches.
        :return: The fused ``(covariances, weights)``, with one branch per group.
        :rtype: Tuple[TensorLike, TensorLike]
        :raises ValueError: if ``weights`` is not a non-empty square weight matrix, if ``groups`` is
            not a partition of the branches, or if ``covariances`` does not carry one entry per
            branch.
        """
        weights_tensor = _validated_weights(weights)
        n_branches = int(weights_tensor.shape[0])
        groups = _validated_groups(groups, n_branches)
        # Read the leading axis off the shape rather than with len(), which raises TypeError rather
        # than the documented ValueError for an input that has no first axis at all.
        covariance_shape = tuple(getattr(covariances, "shape", ()))
        if len(covariance_shape) == 0 or covariance_shape[0] != n_branches:
            raise ValueError(
                f"covariances must carry one entry per branch along its leading axis, got shape "
                f"{covariance_shape} for {n_branches} branches."
            )
        representatives = [group[0] for group in groups]
        if all(len(group) == 1 for group in groups):
            return covariances, weights
        device = weights_tensor.device
        # Each column of the (chi, n_groups) incidence matrix sums the members of one group, so
        # incidence^T W incidence performs both the row and the column addition in one contraction.
        incidence = torch.zeros(n_branches, len(groups), dtype=weights_tensor.dtype, device=device)
        for group_index, group in enumerate(groups):
            members = torch.as_tensor(group, dtype=torch.long, device=device)
            incidence[members, group_index] = 1
        fused = torch.einsum("ag,ab...,bh->gh...", incidence.conj(), weights_tensor, incidence)
        return covariances[representatives], fused

    @staticmethod
    def gram_matrix(weights: TensorLike) -> torch.Tensor:
        r"""Normalized Gram :math:`|W_{\alpha\beta}| / \sqrt{W_{\alpha\alpha} W_{\beta\beta}}`.

        Detached from autograd (the merge decision is discrete) and reduced by the minimum over any
        batch axes. Branches with a vanishing population get a zero row/column, so they are never
        merged; pruning is :class:`~matchcake.devices.branch_tensor.rank_budget.RankBudget`'s job.

        :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
        :return: Real tensor of shape ``(chi, chi)`` with entries in ``[0, 1]``.
        :rtype: torch.Tensor
        :raises ValueError: if ``weights`` is not a non-empty square weight matrix.
        """
        weights_tensor = _validated_weights(weights).detach()
        n_branches = int(weights_tensor.shape[0])
        index = torch.arange(n_branches, device=weights_tensor.device)
        populations = torch.abs(weights_tensor[index, index])  # (chi, ...)
        norm = torch.sqrt(populations[:, None] * populations[None, :])
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        gram = torch.where(norm > 0, torch.abs(weights_tensor) / safe_norm, torch.zeros_like(norm))
        return gram.reshape(n_branches, n_branches, -1).amin(dim=-1)

    def __init__(self, collinear_th: float = EXACT_COLLINEAR_TOL):
        if not 0.0 <= collinear_th < 1.0:
            raise ValueError(
                f"collinear_th must be non-negative and below 1, got {collinear_th}. From 1 onwards "
                f"the threshold 1 - collinear_th is non-positive, which every Gram entry clears, so "
                f"orthogonal branches would be fused."
            )
        self.collinear_th = float(collinear_th)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(collinear_th={self.collinear_th})"

    def count_approximate_merges(self, weights: TensorLike, groups: Sequence[Sequence[int]]) -> int:
        """Number of fused branches whose collinearity was only approximate.

        A branch fused into its group representative counts as approximate when its Gram deficit
        against that representative exceeds :data:`EXACT_COLLINEAR_TOL`; those are the merges that
        make a run inexact.

        :param weights: The weight matrix the groups were derived from.
        :param groups: The groups returned by :meth:`merge_groups`.
        :return: Count of approximate merges.
        :rtype: int
        :raises ValueError: if ``weights`` is not a non-empty square weight matrix, or if ``groups``
            is not a partition of the branches.
        """
        gram = self.gram_matrix(weights)
        groups = _validated_groups(groups, int(gram.shape[0]))
        return sum(
            int(float(1.0 - gram[group[0], member]) > EXACT_COLLINEAR_TOL) for group in groups for member in group[1:]
        )

    def merge_groups(self, weights: TensorLike) -> List[List[int]]:
        r"""Partition the branches into groups of mutually collinear ones.

        Greedy single pass in index order: each branch joins the first earlier group whose
        representative it is collinear with, otherwise it starts its own group. Collinearity is
        transitive up to :math:`\varepsilon_{\mathrm{col}}`.

        :param weights: Complex weight matrix of shape ``(chi, chi)`` or ``(chi, chi, ...)``.
        :return: List of groups, each a list of branch indices with the representative first.
        :rtype: List[List[int]]
        """
        gram = self.gram_matrix(weights)
        threshold = 1.0 - self.collinear_th
        groups: List[List[int]] = []
        for branch in range(int(gram.shape[0])):
            for group in groups:
                if float(gram[group[0], branch]) >= threshold:
                    group.append(branch)
                    break
            else:
                groups.append([branch])
        return groups
