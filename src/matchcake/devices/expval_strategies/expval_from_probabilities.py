from typing import Union

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import pauli_sentence, pauli_word_to_string

from ...observables.batch_hamiltonian import BatchHamiltonian
from ...typing import TensorLike
from ...utils import get_eigvals_on_z_basis
from .expval_strategy import ExpvalStrategy


class ExpvalFromProbabilitiesStrategy(ExpvalStrategy):
    NAME: str = "ExpvalFromProbabilities"

    def __call__(
        self, state_prep_op: Union[qml.StatePrep, qml.BasisState], observable: Operator, **kwargs
    ) -> TensorLike:
        if not self.can_execute(state_prep_op, observable):
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")  # pragma: no cover
        assert "prob" in kwargs, "The probabilities 'prob' must be provided as a keyword argument."
        prob = kwargs["prob"]
        if isinstance(observable, BatchHamiltonian):
            eigvals_on_z_basis = observable.eigvals_on_z_basis()
        else:
            eigvals_on_z_basis = get_eigvals_on_z_basis(observable)
        return qml.math.einsum("...i,...i->...", prob, eigvals_on_z_basis)

    def can_execute(
        self,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
    ) -> bool:
        pauli_kinds = [pauli_word_to_string(op) for op in pauli_sentence(observable)]
        return all((len(set(p) - {"Z", "I"}) == 0) for p in pauli_kinds)
