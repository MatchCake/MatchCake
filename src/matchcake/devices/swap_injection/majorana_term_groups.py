from typing import Dict, List, Optional, Tuple

import torch
from pennylane.operation import Operator, TermsUndefinedError
from pennylane.pauli import pauli_word_to_string

from ...utils import JordanWigner


class MajoranaTermGroups:
    r"""Parsed Majorana term structure of a Pauli-sum observable, grouped by support size and cached on the observable.

    Every branch evaluation of the same Hamiltonian re-derives the identical term structure (Pauli string,
    Majorana support, Wick phase). The parsed structure depends only on ``(observable, wires, marker)``, so it
    is computed once by
    :meth:`from_observable` and cached as an attribute on the observable. Terms are grouped by the size of their
    Majorana support: all submatrices of one group share a shape, so a whole group is evaluated as a single
    batched Pfaffian call along the Pfaffian batch dimension (the same device-controlled batching direction as
    :class:`~matchcake.observables.batch_hamiltonian.BatchHamiltonian`), dropping the kernel-dispatch count from
    the number of terms to the number of distinct support sizes. Parity-odd terms vanish on the basis path
    (``marker=None``) and are lifted with the marker index otherwise.

    :param identity_weight: summed weight of the rank-0 (identity) terms, whose Pfaffian grid is all ones.
    :param groups: one entry per support size: ``(size, index_tensor, weight_tensor)`` with ``index_tensor`` of
        shape ``(n_terms, size)`` (long) and ``weight_tensor`` of shape ``(n_terms,)`` (complex128, the
        coefficient times the Jordan-Wigner sign times the Wick phase of each term).
    """

    CACHE_ATTRIBUTE = "_mc_majorana_term_groups"

    def __init__(self, identity_weight: complex, groups: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> None:
        self.identity_weight = identity_weight
        self.groups = groups

    @classmethod
    def from_observable(cls, observable: Operator, wires: List, marker: Optional[int]) -> "MajoranaTermGroups":
        """Parse (or re-serve from the observable's cache) the grouped Majorana term structure.

        :param observable: A Pauli-decomposable observable (exposing ``observable.terms()``).
        :param wires: Device wire labels in qubit order.
        :param marker: Parity-marker index ``D - 1`` on the lifted path, or ``None`` on the basis path.
        :return: the grouped term structure.
        :rtype: MajoranaTermGroups
        """
        key = (tuple(wires), None if marker is None else int(marker))
        cache: Optional[Dict] = getattr(observable, cls.CACHE_ATTRIBUTE, None)
        if cache is not None and key in cache:
            return cache[key]
        instance = cls._parse(observable, list(wires), marker)
        try:
            if cache is None:
                setattr(observable, cls.CACHE_ATTRIBUTE, {key: instance})
            else:
                cache[key] = instance
        except Exception:  # an observable that forbids attribute assignment simply goes uncached
            pass
        return instance

    @classmethod
    def _parse(cls, observable: Operator, wires: List, marker: Optional[int]) -> "MajoranaTermGroups":
        n_qubits = len(wires)
        jordan_wigner = JordanWigner(n_qubits)
        wire_map = {wire: index for index, wire in enumerate(wires)}
        try:
            coefficients, operators = observable.terms()
        except TermsUndefinedError:
            coefficients, operators = [1.0], [observable]  # a bare Pauli word has no terms() decomposition
        identity_weight = 0.0 + 0.0j
        grouped: Dict[int, List[Tuple[List[int], complex]]] = {}
        for coefficient, operator in zip(coefficients, operators):
            pauli_str = pauli_word_to_string(operator, wire_map=wire_map)
            support, kappa = jordan_wigner.pauli_to_majorana(pauli_str, wires)
            rank = len(support)
            if rank % 2 == 0:
                indices, wick_exponent = list(support), rank // 2
            else:
                if marker is None:
                    continue  # parity-odd term vanishes on the (parity-even) basis path
                indices, wick_exponent = list(support) + [marker], (rank + 1) // 2
            phase = (1j) ** (-wick_exponent)
            coefficient_value = complex(coefficient.item() if isinstance(coefficient, torch.Tensor) else coefficient)
            weight = coefficient_value * kappa * phase
            if indices:
                grouped.setdefault(len(indices), []).append((indices, weight))
            else:
                identity_weight += weight
        groups = [
            (
                size,
                torch.as_tensor([member[0] for member in members], dtype=torch.long),
                torch.as_tensor([member[1] for member in members], dtype=torch.complex128),
            )
            for size, members in grouped.items()
        ]
        return cls(identity_weight, groups)
