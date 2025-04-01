from typing import Optional, Type, List, Sequence, Union, Any

import numpy as np
import pennylane as qml

from .random_generator import RandomOperationsGenerator

from ..operations import (
    MatchgateOperation,
    fRXX,
    fSWAP,
    fRZZ,
    fH,
    fRYY,
    FermionicSuperposition,
)


class RandomParametrizeGenerator(RandomOperationsGenerator):
    pass