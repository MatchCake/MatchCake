from copy import deepcopy
from typing import Sequence, Union

import pennylane as qml
import torch
from pennylane.operation import Operator, TermsUndefinedError
from pennylane.ops.op_math import SProd

from ...typing import TensorLike
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
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
        **kwargs,
    ) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")
        splits = self.split(state_prep_op, observable)
        out = sum(
            sum(op.scalar * strategy(state_prep_op, op, **self._fix_kwargs_prob(op, **kwargs)) for op in split)
            for split, strategy in zip(splits, self.strategies)
        )
        return out

    def split(
        self,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
    ) -> Sequence[Sequence[SProd]]:
        hamiltonian = self._format_observable(observable)
        splits = [
            [SProd(c, op) for c, op in zip(*hamiltonian.terms()) if strategy.can_execute(state_prep_op, op)]
            for strategy in self.strategies
        ]
        return splits

    def can_execute(
        self,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
    ) -> bool:
        hamiltonian = self._format_observable(observable)
        return all(any(s.can_execute(state_prep_op, op) for s in self.strategies) for op in hamiltonian.ops)

    def _fix_kwargs_prob(self, op: SProd, **kwargs):
        new_kwargs = {**kwargs}
        if "prob_func" in new_kwargs:
            new_kwargs["prob"] = new_kwargs["prob_func"](op.wires)
        return new_kwargs

    @staticmethod
    def _format_observable(observable):
        try:
            terms = observable.terms()
        except TermsUndefinedError:
            terms = torch.ones((1,)), [observable]
        return qml.Hamiltonian(*terms)

    @property
    def strategies(self) -> Sequence[ExpvalStrategy]:
        return self._strategies
