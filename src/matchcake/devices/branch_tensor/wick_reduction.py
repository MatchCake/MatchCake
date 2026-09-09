from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from matchcake.typing import TensorLike

from ...utils import signed_pfaffian_complex


class WickReduction:
    r"""Two-state Wick reduction of Majorana monomials against a transition covariance.

    Evaluates weighted sums of the form

    .. math::
        \sum_t \lambda_t\, i^{-w_t/2}\, \mathrm{Pf}\bigl(\Gamma|_{S_t}\bigr),

    where each :math:`S_t` is a sorted, duplicate-free tuple of Majorana mode indices of even length
    :math:`w_t`, :math:`\lambda_t` is its complex coefficient, and :math:`\Gamma` is a complex
    antisymmetric transition covariance. Each term is the matrix element of the Majorana monomial
    :math:`c_{S_t}` between two fermionic Gaussian states, divided by their overlap.

    :math:`\Gamma` may carry arbitrary leading batch axes, shape ``(..., D, D)``; passing the full
    ``(chi, chi, ..., D, D)`` branch-pair grid evaluates every branch pair at once.

    :meth:`sum_monomials` groups the monomials by support size and issues one batched Pfaffian call
    per distinct size, so the number of kernel dispatches is the number of distinct support sizes
    instead of the number of terms.

    Supports of odd length are rejected: Gaussian states of opposite fermion parity are exactly
    orthogonal, so the overlap-normalized matrix element is an indeterminate :math:`0/0`.

    References: fermionic linear optics and the two-state Wick theorem (Bravyi,
    arXiv:quant-ph/0404180).
    """

    @staticmethod
    def element(support: Tuple[int, ...], transition_covariance: TensorLike) -> TensorLike:
        r"""Two-state Wick value :math:`i^{-w/2} \mathrm{Pf}(\Gamma|_S)` of one sorted monomial.

        The empty support gives the constant 1, broadcast over the leading axes of
        ``transition_covariance``.

        :param support: Sorted, duplicate-free Majorana mode indices ``S``.
        :param transition_covariance: Complex antisymmetric ``(..., D, D)`` transition covariance.
        :return: Complex tensor of shape ``(...)`` (the leading axes of ``transition_covariance``).
        :rtype: TensorLike
        :raises NotImplementedError: if ``len(support)`` is odd.
        """
        weight = len(support)
        WickReduction._assert_even_support(support)
        batch_shape = tuple(transition_covariance.shape[:-2])
        if weight == 0:
            return torch.ones(batch_shape, dtype=transition_covariance.dtype, device=transition_covariance.device)
        index = torch.as_tensor(support, dtype=torch.long, device=transition_covariance.device)
        submatrix = transition_covariance.index_select(-2, index).index_select(-1, index)  # (..., w, w)
        return WickReduction.phase(weight) * signed_pfaffian_complex(submatrix)

    @staticmethod
    def phase(weight: int) -> complex:
        r"""The uniform Wick phase :math:`i^{-w/2}` for a monomial of Majorana weight ``w``.

        :param weight: Even number of Majorana factors.
        :return: The phase.
        :rtype: complex
        """
        return (1j) ** (-(weight // 2))

    @staticmethod
    def sum_monomials(
        supports: Sequence[Tuple[int, ...]],
        coefficients: Sequence[complex],
        transition_covariance: TensorLike,
        chunk_size: Optional[int] = None,
    ) -> TensorLike:
        r"""Weighted sum :math:`\sum_t \lambda_t\, i^{-w_t/2} \mathrm{Pf}(\Gamma|_{S_t})`, batched by size.

        Monomials are grouped by support size and each group is evaluated by one batched Pfaffian
        call. An empty support contributes its coefficient as a constant, with no Pfaffian.

        Repeated supports are summed rather than merged, so the same support may appear more than once
        with different coefficients.

        :param supports: Sorted, duplicate-free Majorana mode index tuples, one per monomial.
        :param coefficients: Complex coefficient of each monomial, index-aligned with ``supports``.
        :param transition_covariance: Complex antisymmetric ``(..., D, D)`` transition covariance,
            typically the full ``(chi, chi, ..., D, D)`` branch-pair grid.
        :param chunk_size: Maximum number of matrices reduced per batched Pfaffian call, forwarded to
            bound peak memory. Defaults to ``None`` (no chunking).
        :return: Complex tensor of shape ``(...)`` (the leading axes of ``transition_covariance``).
        :rtype: TensorLike
        :raises ValueError: if ``supports`` and ``coefficients`` have different lengths.
        :raises NotImplementedError: if any support has odd length.
        """
        if len(supports) != len(coefficients):
            raise ValueError(
                f"sum_monomials got {len(supports)} supports and {len(coefficients)} coefficients; "
                f"they must be index-aligned."
            )
        dtype = transition_covariance.dtype
        device = transition_covariance.device
        batch_shape = tuple(transition_covariance.shape[:-2])
        total = torch.zeros(batch_shape, dtype=dtype, device=device)

        by_size: Dict[int, List[int]] = defaultdict(list)
        for position, support in enumerate(supports):
            WickReduction._assert_even_support(support)
            by_size[len(support)].append(position)

        for size, positions in sorted(by_size.items()):
            group_coefficients = torch.as_tensor(
                [complex(coefficients[position]) for position in positions], dtype=dtype, device=device
            )  # (n_terms,)
            if size == 0:
                total = total + group_coefficients.sum()
                continue
            index_tensor = torch.as_tensor(
                [supports[position] for position in positions], dtype=torch.long, device=device
            )  # (n_terms, size)
            rows = index_tensor[:, :, None].expand(-1, size, size)
            cols = index_tensor[:, None, :].expand(-1, size, size)
            submatrices = transition_covariance[..., rows, cols]  # (..., n_terms, size, size)
            pfaffian_kwargs = {} if chunk_size is None else {"chunk_size": chunk_size}
            pfaffians = signed_pfaffian_complex(submatrices, **pfaffian_kwargs)  # (..., n_terms)
            total = total + (pfaffians * group_coefficients * WickReduction.phase(size)).sum(-1)
        return total

    @staticmethod
    def _assert_even_support(support: Tuple[int, ...]) -> None:
        """Validate a Majorana support.

        A negative mode index is rejected rather than interpreted as an offset from the end, so that a
        malformed support raises instead of silently selecting the wrong modes.

        :param support: The monomial's Majorana mode indices.
        :return: None
        :raises NotImplementedError: if the support has odd length.
        :raises ValueError: if any mode index is negative.
        """
        if any(mode < 0 for mode in support):
            raise ValueError(
                f"Majorana support {support} contains a negative mode index. Supports must be sorted, "
                f"duplicate-free, and within the transition covariance's dimension."
            )
        if len(support) % 2 == 0:
            return
        raise NotImplementedError(
            f"Odd Majorana support {support}: the two-state Wick rule is defined for even supports "
            f"only. Gaussian states of opposite fermion parity are exactly orthogonal, so the "
            f"overlap-normalized matrix element of an odd monomial is an indeterminate 0/0. Only "
            f"parity-even operators are in scope."
        )
