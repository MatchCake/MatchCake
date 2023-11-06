import numpy as np
import pytest

from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams
)
from msim.utils import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_I,
)
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)

