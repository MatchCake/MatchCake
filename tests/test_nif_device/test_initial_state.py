from functools import partial
from typing import Tuple

import numpy as np
import pennylane as qml
import pytest

from matchcake import Matchgate, MatchgateOperation, NonInteractingFermionicDevice
from matchcake import matchgate_parameter_sets as mps
from matchcake import utils
from matchcake.utils import PAULI_X, PAULI_Z

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from . import devices_init

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "initial_binary_state",
    [np.random.randint(0, 2, size=n) for n in range(2, 10) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_single_gate_circuit_analytic_probability_lt_vs_es(initial_binary_state):
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    device.apply(qml.BasisState(initial_binary_state, wires=range(len(initial_binary_state))))
    state = device.state
    binary_state = device.binary_state
    initial_state = utils.binary_state_to_state(initial_binary_state)
    np.testing.assert_allclose(
        initial_state,
        state,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_allclose(
        initial_binary_state,
        binary_state,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
