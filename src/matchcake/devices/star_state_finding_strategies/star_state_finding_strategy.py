from abc import ABC, abstractmethod
from typing import Callable, Tuple

import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.wires import Wires


class StarStateFindingStrategy(ABC):
    NAME: str = "StarStateFindingStrategy"

    @abstractmethod
    def __call__(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> Tuple[TensorLike, TensorLike]:
        raise NotImplementedError("This method should be implemented by the subclass.")
