from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from matchcake import MatchgateOperation, NonInteractingFermionicDevice, Matchgate
from matchcake import matchgate_parameter_sets as mps
from matchcake import utils
from functools import partial

from matchcake.utils import PAULI_Z, PAULI_X
from . import devices_init
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "initial_binary_state",
    [
        np.random.randint(0, 2, size=n)
        for n in range(2, 10)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_gate_circuit_analytic_probability_lt_vs_es(initial_binary_state):
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    device.apply(qml.BasisState(initial_binary_state, wires=range(len(initial_binary_state))))
    state = device.state
    initial_state = utils.binary_state_to_state(initial_binary_state)
    np.testing.assert_allclose(
        initial_state, state,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
