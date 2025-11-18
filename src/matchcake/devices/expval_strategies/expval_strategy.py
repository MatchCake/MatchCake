from typing import Union

import pennylane as qml
from pennylane.operation import Operator

from ...typing import TensorLike


class ExpvalStrategy:
    NAME: str = "ExpvalStrategy"

    def __call__(
        self,
        global_sptm: TensorLike,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
        **kwargs,
    ) -> TensorLike:
        raise NotImplementedError()

    def can_execute(
        self,
        state_prep_op: Union[qml.StatePrep, qml.BasisState],
        observable: Operator,
    ) -> bool:
        raise NotImplementedError()
