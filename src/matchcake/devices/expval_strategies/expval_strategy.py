from typing import Union

import pennylane as qml
from pennylane.operation import Operator, StatePrepBase

from ...typing import TensorLike


class ExpvalStrategy:
    NAME: str = "ExpvalStrategy"

    def __call__(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
        **kwargs,
    ) -> TensorLike:
        raise NotImplementedError()

    def can_execute(
        self,
        state_prep_op: StatePrepBase,
        observable: Operator,
    ) -> bool:
        raise NotImplementedError()
