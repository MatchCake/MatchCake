from ...typing import TensorLike
import pennylane as qml


class ExpvalStrategy:
    NAME: str = "ExpvalStrategy"

    def __call__(
            self,
            global_sptm: TensorLike,
            state_prep_op: qml.StatePrep,
            observable: qml.Hamiltonian,
            **kwargs
    ) -> TensorLike:
        raise NotImplementedError()

