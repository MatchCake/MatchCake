from typing import Union

import pennylane as qml
import torch
from pennylane.operation import Operator, StatePrepBase, TermsUndefinedError
from pennylane.ops.op_math import Prod
from pennylane.ops.qubit import Projector
from pennylane.pauli import pauli_sentence, pauli_word_to_string

from ...observables.batch_hamiltonian import BatchHamiltonian
from ...typing import TensorLike
from ...utils import get_eigvals_on_z_basis
from .expval_strategy import ExpvalStrategy


class ExpvalFromProbabilitiesStrategy(ExpvalStrategy):
    NAME: str = "ExpvalFromProbabilities"

    def __call__(self, state_prep_op: StatePrepBase, observable: Operator, **kwargs) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")  # pragma: no cover
        hamiltonian = self._format_observable(observable)
        probs = self.gather_probs(hamiltonian, **kwargs)
        if isinstance(observable, BatchHamiltonian):
            eigvals_on_z_basis = observable.eigvals_on_z_basis()
            return qml.math.einsum("k,...ki,...ki->...k", observable.coeffs, probs, eigvals_on_z_basis)
        eigvals_on_z_basis = qml.math.stack([get_eigvals_on_z_basis(t) for c, t in zip(*hamiltonian.terms())])
        return qml.math.einsum("k,...ki,ki->...", hamiltonian.coeffs, probs, eigvals_on_z_basis)

    def gather_probs(self, hamiltonian, **kwargs) -> TensorLike:
        assert any(
            [s in kwargs for s in ["prob", "prob_func"]]
        ), "Either 'prob' or 'prob_func' must be provided as a keyword argument."
        if "prob" in kwargs:
            return kwargs["prob"]
        assert "prob_func" in kwargs, "The probability function 'prob_func' must be provided as a keyword argument."
        assert callable(kwargs["prob_func"]), "The 'prob_func' keyword argument must be a callable function."
        prob_func = kwargs["prob_func"]
        probs = prob_func([t.wires for c, t in zip(*hamiltonian.terms())])
        return probs

    def can_execute(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
    ) -> bool:
        if isinstance(observable, (Projector,)):
            return False
        pauli_kinds = [pauli_word_to_string(op) for op in pauli_sentence(observable)]
        return all((len(set(p) - {"Z", "I"}) == 0) for p in pauli_kinds)

    @staticmethod
    def _format_observable(observable):
        if isinstance(observable, BatchHamiltonian):
            return observable
        if isinstance(observable, Prod):
            return qml.Hamiltonian([1.0], [observable])
        try:
            terms = observable.terms()
        except TermsUndefinedError:
            terms = torch.ones((1,)), [observable]
        hamiltonian = qml.Hamiltonian(*terms)
        return hamiltonian
