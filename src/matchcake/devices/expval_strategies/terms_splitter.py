from typing import List, Sequence

import pennylane as qml
from pennylane.operation import Operator, StatePrepBase
from pennylane.ops.op_math import SProd

from ...typing import TensorLike
from ...utils import live_pauli_terms, stacked_coefficients
from .expval_strategy import ExpvalStrategy


class TermsSplitter(ExpvalStrategy):
    def __init__(
        self,
        strategies: Sequence[ExpvalStrategy],
    ):
        self._strategies = strategies
        self.NAME = "-".join(s.NAME for s in strategies)

    def __call__(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
        **kwargs,
    ) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")
        splits = self.split(state_prep_op, observable)
        out = sum(
            sum(op.scalar * strategy(state_prep_op, op.base, **kwargs) for op in split)
            for split, strategy in zip(splits, self.strategies)
        )
        return out

    def split(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
    ) -> Sequence[Sequence[SProd]]:
        hamiltonian = self._format_observable(observable)
        splits: List[List[SProd]] = [[] for _ in self.strategies]
        for coefficient, operator in zip(*hamiltonian.terms()):
            for split, strategy in zip(splits, self.strategies):
                if strategy.can_execute(state_prep_op, operator):
                    split.append(SProd(coefficient, operator))
                    break
        return splits

    def can_execute(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
    ) -> bool:
        hamiltonian = self._format_observable(observable)
        return all(any(s.can_execute(state_prep_op, op) for s in self.strategies) for op in hamiltonian.ops)

    @staticmethod
    def _format_observable(observable):
        coefficients, operators = live_pauli_terms(observable, keep_non_pauli=True)
        if not operators:
            return qml.Hamiltonian([], [])
        return qml.Hamiltonian(stacked_coefficients(list(coefficients)), operators)

    @property
    def strategies(self) -> Sequence[ExpvalStrategy]:
        return self._strategies
