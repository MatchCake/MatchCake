from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml

from ..operations import (
    FermionicSuperposition,
    MatchgateOperation,
    fH,
    fRXX,
    fRYY,
    fRZZ,
    fSWAP,
)
from .random_generator import RandomOperationsGenerator


class RandomParametrizeGenerator(RandomOperationsGenerator):
    pass
