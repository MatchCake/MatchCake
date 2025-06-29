from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import pennylane as qml
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from matchcake.utils.math import random_index
from matchcake.utils.torch_utils import to_numpy


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
