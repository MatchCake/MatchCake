from typing import Any, List, Optional, Sequence, Type, Union

import numpy as np
import pennylane as qml

from .random_generator import RandomOperationsGenerator
from .random_matchgate_circuits import (
    RandomMatchgateHaarOperationsGenerator,
    RandomMatchgateOperationsGenerator,
)
from .random_sptm_circuits import (
    RandomSptmHaarOperationsGenerator,
    RandomSptmOperationsGenerator,
    random_sptm_operations_generator,
)
