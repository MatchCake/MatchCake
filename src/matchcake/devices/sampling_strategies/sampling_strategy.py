from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires
import pennylane as qml

from matchcake.utils.math import random_index
from matchcake.utils.torch_utils import to_numpy


class SamplingStrategy(ABC):
    NAME: str = "SamplingStrategy"

    @abstractmethod
    def generate_samples(
            self,
            device: qml.QubitDevice,
            state_prob_func: Callable[[TensorLike, Wires], TensorLike],
            **kwargs
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def batch_generate_samples(
            self,
            device: qml.QubitDevice,
            states_prob_func: Callable[[TensorLike, Wires], TensorLike],
            **kwargs
    ) -> TensorLike:
        raise NotImplementedError("This method should be implemented by the subclass.")
