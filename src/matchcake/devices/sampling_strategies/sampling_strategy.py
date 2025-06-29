from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pennylane as qml
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires


class SamplingStrategy(ABC):
    NAME: str = "SamplingStrategy"

    @abstractmethod
    def generate_samples(
        self,
        device: qml.devices.QubitDevice,
        state_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def batch_generate_samples(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Callable[[TensorLike, Wires], TensorLike],
        **kwargs,
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")
