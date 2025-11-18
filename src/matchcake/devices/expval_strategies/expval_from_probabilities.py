from typing import Union

import pennylane as qml
from pennylane.operation import Operator

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
            raise ValueError(f"Cannot execute {self.NAME} strategy for {observable}.")
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
        # TODO: Need to check if the observable is diagonal on the Z basis.
        return True
