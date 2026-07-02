from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.wires import Wires

from ...operations.single_particle_transition_matrices.sptm_fswap import SptmCompZX
from ...typing import TensorLike
from ...utils._pfaffian import signed_pfaffian_complex
from ...utils.math import convert_and_cast_like
from .branch_observables import transition_cov
from .lift import lift_sptm

# Lambda_occ: covariance of the occupied configuration |1>_j |1>_k; (Lambda_occ)_{01} = (Lambda_occ)_{23} = +1.
_OCCUPIED_BLOCK = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=torch.float64)

# Branch pairs with |<phi_a|phi_b>|^2 = 2^{-D/2} |Pf(Lambda_a + Lambda_b)| below this are treated as
# orthogonal: the overlap-normalized observables become ill-conditioned there (0 x inf in
# swap_injection_theory.md section 10), so the state is flagged degenerate and observables reroute
# to the overlap-free string engine. At this threshold the branch-path conditioning error is still
# ~1e-12, so the switch is seamless on both sides.
DEGENERATE_OVERLAP_TOL = 1e-4


def condition_occupied(covariance: TensorLike, j: int, k: int) -> TensorLike:
    r"""Covariance of branches with modes ``j, k`` projected to occupied (swap_injection_theory.md eq 17), batched.

    With ``S4 = {2j, 2j+1, 2k, 2k+1}`` and ``R`` the rest (every Majorana mode not pinned; the two
    ancilla rows/cols stay in ``R`` on the lifted path), the block split ``cov = [[A, B], [-B^T, C]]``
    gives, with ``Lambda_occ`` the covariance of the occupied configuration ``|1>_j |1>_k``,

    .. math::
        \Lambda'|_{S4} = \Lambda_{\mathrm{occ}}, \quad \Lambda'|_{S4, R} = 0, \quad
        \Lambda'|_R = C + B^T (A + \Lambda_{\mathrm{occ}})^{-1} B.

    The last term is the fermionic Schur-complement back-action of the projection (the analogue of
    classical Gaussian conditioning); see swap_injection_theory.md section 7 for the derivation.

    Acts on any leading batch (e.g. the whole ``(chi, ..., D, D)`` branch tensor) at once.

    :param covariance: Real antisymmetric covariance(s) of shape ``(..., D, D)``.
    :param j: First qubit index.
    :param k: Second qubit index.
    :return: Conditioned covariance(s) of shape ``(..., D, D)``, on the backend/dtype of ``covariance``.
    :rtype: TensorLike
    """
    covariance_t = torch.as_tensor(
        qml.math.toarray(covariance) if not isinstance(covariance, torch.Tensor) else covariance
    )
    if not torch.is_floating_point(covariance_t):
        covariance_t = covariance_t.to(torch.float64)
    dim = covariance_t.shape[-1]
    occupied_modes = [2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
    rest_modes = [mode for mode in range(dim) if mode not in occupied_modes]
    occupied_index = torch.as_tensor(occupied_modes, dtype=torch.long)
    rest_index = torch.as_tensor(rest_modes, dtype=torch.long)
    occupied_block = _OCCUPIED_BLOCK.to(dtype=covariance_t.dtype, device=covariance_t.device)

    block_pinned = covariance_t.index_select(-2, occupied_index).index_select(-1, occupied_index)
    block_cross = covariance_t.index_select(-2, occupied_index).index_select(-1, rest_index)
    block_rest = covariance_t.index_select(-2, rest_index).index_select(-1, rest_index)
    # block_pinned + occupied_block is invertible whenever the projected branch has nonzero weight;
    # it is singular exactly when the branch vanishes (e.g. swapping an ancilla pinned to |0>, q = 0),
    # in which case the conditioned covariance is irrelevant because the branch is pruned right after.
    # The pseudo-inverse gives a finite (pruned) result instead of raising.
    conditioned_rest = (
        block_rest + block_cross.transpose(-1, -2) @ torch.linalg.pinv(block_pinned + occupied_block) @ block_cross
    )

    conditioned = torch.zeros_like(covariance_t)
    conditioned[..., occupied_index[:, None], occupied_index[None, :]] = occupied_block
    conditioned[..., rest_index[:, None], rest_index[None, :]] = conditioned_rest
    return convert_and_cast_like(conditioned, covariance)


class SwapBranchState:
    r"""Sum-of-Gaussians state ``(cov, W)`` for a matchgate + SWAP circuit (swap_injection_theory.md eqs 7-8).

    Holds the covariance tensor ``cov`` of shape ``(chi, D, D)`` and the Hermitian weight matrix
    ``W`` of shape ``(chi, chi)``. Matchgate layers act on every branch by the SPTM sandwich
    (:meth:`apply_matchgate_sptm`, eq 4); a genuine SWAP branches the state (:meth:`apply_swap`,
    section 6), doubling ``chi`` before pruning.

    Every branching also refreshes the per-pair bookkeeping behind :attr:`string_pair_mask`: a pair
    of branches whose overlap is (near-)zero, or whose weight entry was propagated through such a
    pair (tainted lineage), cannot be evaluated by the overlap-normalized formulas (they hit
    ``0 x inf``, swap_injection_theory.md section 10). The device evaluates the masked pairs
    overlap-free from the per-branch projector ``histories`` and keeps the fast path for the rest;
    :attr:`degenerate` is True as soon as any pair is masked.

    :param branch_covariances: Real covariance tensor of shape ``(chi, D, D)``.
    :param weights: Complex Hermitian weight matrix ``W`` of shape ``(chi, chi)``.
    :param lifted: Whether the covariances live in the even ``(2n+2)`` lift.
    :param marker: Parity-marker index (``D - 1 = 2n+1`` when lifted, else ``None``).
    """

    def __init__(
        self,
        branch_covariances: TensorLike,
        weights: TensorLike,
        lifted: bool = False,
        marker: Optional[int] = None,
    ):
        self.cov = branch_covariances
        self.weights = weights
        self.lifted = lifted
        default_marker = qml.math.shape(branch_covariances)[-1] - 1 if lifted else None
        self.marker = marker if marker is not None else default_marker
        n_branches = int(qml.math.shape(branch_covariances)[0])
        self.histories: List[Tuple[int, ...]] = [() for _ in range(n_branches)]
        self._pair_overlap_sq = np.ones((n_branches, n_branches))
        self._pair_needs_string = np.zeros((n_branches, n_branches), dtype=bool)

    def apply_matchgate_sptm(self, sptm: TensorLike) -> "SwapBranchState":
        r"""Evolve every branch by a matchgate SPTM: ``cov[a] <- Q'^T cov[a] Q'`` (swap_injection_theory.md eq 4).

        ``Q'`` is ``sptm`` on the basis path and ``sptm (+) I_2`` (via :func:`lift_sptm`) on the
        lift; the weights are unchanged. ``sptm`` may carry leading broadcast dims.

        :param sptm: Physical ``(..., 2n, 2n)`` SPTM (or the already-lifted ``(..., D, D)`` SPTM).
        :return: ``self`` (mutated in place).
        :rtype: SwapBranchState
        """
        lifted_sptm = sptm
        if self.lifted and int(qml.math.shape(sptm)[-1]) == self.dim - 2:
            lifted_sptm = lift_sptm(sptm)
        lifted_sptm = convert_and_cast_like(lifted_sptm, self.cov)
        sptm_transposed = qml.math.swapaxes(lifted_sptm, -1, -2)
        self.cov = qml.math.einsum("...ij,a...jk->a...ik", sptm_transposed, self.cov)
        self.cov = qml.math.einsum("a...ij,...jk->a...ik", self.cov, lifted_sptm)
        return self

    def apply_cz(self, j: int, k: int, tol: float = 1e-12) -> "SwapBranchState":
        r"""Apply a genuine ``CZ(j, k) = 1 - 2 n_j n_k`` (swap_injection_theory.md section 2).

        ``CZ`` is the non-Gaussian (quartic) factor of a ``SWAP``: like a ``SWAP`` it branches each
        Gaussian into an unchanged "type-0" branch and an occupation-conditioned "type-1" branch, but
        unlike a ``SWAP`` it does not apply the trailing ``fSWAP`` matchgate. The branching, weight
        block update (eq 19), and pruning are shared with :meth:`apply_swap` via
        :meth:`_branch_on_occupation`.

        :param j: First qubit index.
        :param k: Second qubit index.
        :param tol: Branches with ``|W_{aa}| < tol`` are pruned.
        :return: ``self`` (mutated in place).
        :rtype: SwapBranchState
        """
        return self._branch_on_occupation(j, k, tol=tol)

    def apply_swap(self, j: int, k: int, tol: float = 1e-12) -> "SwapBranchState":
        r"""Apply a genuine ``SWAP(j, k) = fSWAP(j, k) . (1 - 2 n_j n_k)`` (swap_injection_theory.md section 6).

        Splits each branch into the fSWAP-only branch and the CZ-projected (occupation-conditioned)
        branch, updates the weight matrix by the block rule (eq 19), prunes vanished branches, then
        applies the fSWAP SPTM to every surviving branch. The branching step is shared with
        :meth:`apply_cz` (a ``SWAP`` is a ``CZ`` followed by the matchgate ``fSWAP``).

        Only adjacent wires are accepted: ``SWAP = fSWAP . CZ`` holds only there. A non-adjacent
        qubit swap also carries the parity phase of every crossed mode,
        ``SWAP_{jk} = M_{jk} CZ_{jk} prod_{j<l<k} CZ_{jl} CZ_{kl}``, which the device decomposes
        into individual :meth:`apply_cz` calls plus the mode-swap SPTM.

        :param j: First qubit index.
        :param k: Second qubit index.
        :param tol: Branches with ``|W_{aa}| < tol`` are pruned.
        :return: ``self`` (mutated in place).
        :rtype: SwapBranchState
        :raises ValueError: if ``j`` and ``k`` are not adjacent.
        """
        if abs(j - k) != 1:
            raise ValueError(
                f"apply_swap requires adjacent qubits, got ({j}, {k}): a non-adjacent qubit SWAP is "
                f"not fSWAP . CZ (it also carries the parity phase of every crossed mode). Decompose "
                f"it into apply_cz calls plus the mode-swap SPTM, as SwapAugmentedFermionicDevice does."
            )
        self._branch_on_occupation(j, k, tol=tol)

        # Apply the fSWAP matchgate to every surviving branch (step E).
        fswap_sptm = SptmCompZX(wires=[j, k]).pad(Wires(range(self.num_wires))).matrix()
        self.apply_matchgate_sptm(fswap_sptm)
        return self

    def _branch_on_occupation(self, j: int, k: int, tol: float = 1e-12) -> "SwapBranchState":
        r"""Branch each Gaussian on ``1 - 2 n_j n_k`` (swap_injection_theory.md section 7), the shared
        non-Gaussian core of both ``CZ`` and ``SWAP``.

        Doubles ``chi`` into the unchanged "type-0" branches and the occupation-conditioned "type-1"
        branches, updates the weight matrix by the block rule (eq 19), and prunes vanished branches.
        The cross-occupation matrix ``q`` and the conditioning are computed for all branch
        pairs/branches at once (no per-branch Python loop).

        :param j: First qubit index.
        :param k: Second qubit index.
        :param tol: Branches with ``|W_{aa}| < tol`` are pruned.
        :return: ``self`` (mutated in place).
        :rtype: SwapBranchState
        """
        chi = self.chi
        # Pairs already masked before this branching produce unreliable q entries below; their
        # descendants inherit the mask (tainted lineage) and are evaluated overlap-free later.
        parent_mask = self.string_pair_mask  # (chi, chi)
        # Cross occupation matrix q_{ab} (eq 18) over every branch pair, vectorized.
        gamma = transition_cov(self.cov[:, None], self.cov[None, :], marker=self.marker)  # (chi, chi, ..., D, D)
        s4 = torch.as_tensor([2 * j, 2 * j + 1, 2 * k, 2 * k + 1], dtype=torch.long)
        occupied_pfaffian = signed_pfaffian_complex(gamma.index_select(-2, s4).index_select(-1, s4))
        q_matrix = 0.25 * (1.0 + gamma[..., 2 * j, 2 * j + 1] + gamma[..., 2 * k, 2 * k + 1] + occupied_pfaffian)
        q_matrix = convert_and_cast_like(q_matrix, self.weights)  # (chi, chi, ...)

        # New covariances: type-0 (unchanged) then type-1 (occupation-conditioned, all branches at once).
        conditioned = condition_occupied(self.cov, j, k)
        new_cov = qml.math.concatenate([self.cov, conditioned], axis=0)  # (2chi, ..., D, D)

        # Weight block update (eq 19): W_new = [[W, -2 W.*q], [-2 W.*q, 4 W.*q]]. The chi axes are the
        # leading pair; align W's rank to q (append trailing batch axes), then concat along axes 1, 0.
        weights = self.weights
        for _ in range(qml.math.ndim(q_matrix) - qml.math.ndim(weights)):
            weights = weights[..., None]
        weighted_q = weights * q_matrix
        weights_block = weights * qml.math.ones_like(q_matrix)
        top = qml.math.concatenate([weights_block, -2 * weighted_q], axis=1)
        bottom = qml.math.concatenate([-2 * weighted_q, 4 * weighted_q], axis=1)
        new_weights = qml.math.concatenate([top, bottom], axis=0)  # (2chi, 2chi, ...)

        # Prune vanished branches (step D): keep those with |W_new[a, a]| >= tol (max over any batch).
        diagonal = qml.math.stack([qml.math.abs(new_weights[a, a]) for a in range(2 * chi)])
        if isinstance(diagonal, torch.Tensor):
            diagonal = diagonal.detach()  # pruning is a discrete decision; float() on a grad tensor warns
        keep = [a for a in range(2 * chi) if float(qml.math.max(diagonal[a])) >= tol]
        self.cov = new_cov[keep]
        self.weights = new_weights[keep][:, keep]
        new_histories = [history + (0,) for history in self.histories] + [history + (1,) for history in self.histories]
        self.histories = [new_histories[index] for index in keep]
        tiled_mask = np.tile(parent_mask, (2, 2))  # every child pair inherits its parent pair's mask
        self._pair_needs_string = tiled_mask[np.ix_(keep, keep)]
        self._refresh_pair_overlaps()
        return self

    def _refresh_pair_overlaps(self) -> None:
        r"""Recompute the pairwise overlap magnitudes of the surviving branches.

        Pairwise overlaps satisfy ``|<phi_a|phi_b>|^2 = 2^{-D/2} |Pf(cov_a + cov_b)|`` and are
        invariant under matchgates, so refreshing after every branching keeps the matrix valid for
        the whole circuit. Batched covariances are reduced by the minimum over the batch, so a pair
        is masked as soon as any batch element makes it (near-)orthogonal.

        :return: None
        """
        if self.chi < 2:
            self._pair_overlap_sq = np.ones((self.chi, self.chi))
            return
        with torch.no_grad():
            covariances = torch.as_tensor(
                qml.math.toarray(self.cov) if not isinstance(self.cov, torch.Tensor) else self.cov
            )
            pair_pfaffians = signed_pfaffian_complex(covariances[:, None] + covariances[None, :])  # (chi, chi, ...)
            overlap_sq = torch.abs(torch.as_tensor(qml.math.toarray(pair_pfaffians))) * 2.0 ** (-self.dim / 2)
            overlap_sq = overlap_sq.reshape(self.chi, self.chi, -1).amin(dim=-1)
        self._pair_overlap_sq = overlap_sq.cpu().numpy()

    @property
    def chi(self) -> int:
        """Number of Gaussian branches."""
        return int(qml.math.shape(self.cov)[0])

    @property
    def degenerate(self) -> bool:
        """Whether any branch pair must be evaluated overlap-free (see :attr:`string_pair_mask`)."""
        return bool(self.string_pair_mask.any())

    @property
    def dim(self) -> int:
        """Majorana dimension ``D`` of each branch covariance."""
        return int(qml.math.shape(self.cov)[-1])

    @property
    def num_wires(self) -> int:
        """Number of physical qubits ``n``."""
        return (self.dim - 2) // 2 if self.lifted else self.dim // 2

    @property
    def string_pair_mask(self) -> np.ndarray:
        """Boolean ``(chi, chi)`` mask of the branch pairs the overlap-normalized formulas cannot handle.

        A pair is masked when its overlap is below ``DEGENERATE_OVERLAP_TOL`` (the ``0 x inf`` of
        swap_injection_theory.md section 10) or when its weight entry was propagated through such a
        pair at an earlier branching (tainted lineage: the parents' cross occupation ``q`` was
        unreliable, even if the children are no longer orthogonal).

        :return: Boolean mask of shape ``(chi, chi)``.
        :rtype: np.ndarray
        """
        return self._pair_needs_string | (self._pair_overlap_sq < DEGENERATE_OVERLAP_TOL)
