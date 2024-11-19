import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import (
    fSWAP, SptmRzRz, SptmFSwapRzRz,
)
from matchcake.operations import FermionicSuperposition
import pennylane as qml
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)




