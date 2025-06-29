import numpy as np
import pennylane as qml
import psutil
import pytest

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake import matchgate_parameter_sets as mps
from matchcake.devices.probability_strategies import get_probability_strategy

from .. import get_slow_test_mark
from ..configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "initial_binary_state,params,wires,target_binary_state",
    [
        (
            np.random.randint(0, 2, size=n),
            mps.MatchgatePolarParams.random(),
            [0, 1],
            np.random.randint(0, 2, size=n),
        )
        for n in [
            2,
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_single_gate_circuit_probability_lt_vs_es(initial_binary_state, params, wires, target_binary_state):
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=[0, 1]),
    ]
    device.apply(operations)

    device.prob_strategy = get_probability_strategy("LookupTable")
    lt_probs = device.get_state_probability(target_binary_state, wires)
    device.prob_strategy = get_probability_strategy("ExplicitSum")
    es_probs = device.get_state_probability(target_binary_state, wires)
    np.testing.assert_allclose(
        lt_probs,
        es_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "initial_binary_state,params,wires,target_binary_state",
    [
        (
            np.random.randint(0, 2, size=n),
            mps.MatchgatePolarParams.random(),
            [0, 1],
            np.random.randint(0, 2, size=n),
        )
        for n in [
            2,
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_single_gate_circuit_probability_lt_vs_es_mp(initial_binary_state, params, wires, target_binary_state):
    if psutil.cpu_count() < 2:
        pytest.skip("This test requires at least 2 CPUs.")
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state), n_workers=2)
    assert device.n_workers == 2
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=[0, 1]),
    ]
    device.apply(operations)
    device.prob_strategy = get_probability_strategy("LookupTable")
    lt_probs = device.get_state_probability(target_binary_state, wires)
    device.prob_strategy = get_probability_strategy("ExplicitSum")
    es_probs = device.get_state_probability(target_binary_state, wires)
    np.testing.assert_allclose(
        lt_probs,
        es_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
