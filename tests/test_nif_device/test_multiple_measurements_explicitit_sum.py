import numpy as np
import pennylane as qml
import pytest

from msim import MatchgateOperator, NonInteractingFermionicDevice
from msim import matchgate_parameter_sets as mps
from msim import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "initial_binary_string,params,wire",
    [
        ("01", mps.fSWAP, 0),
    ]
)
def test_single_gate_circuit_probability_single_vs_target_specific_cases(initial_binary_string, params, wire):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperator(params, wires=[0, 1])
    ]
    device.apply(operations)
    es_probs = device.compute_probability_using_explicit_sum(wire)
    es_m_probs = np.asarray([
        # device.compute_probability_of_target_using_explicit_sum(wire, target_binary_state="0"),
        device.compute_probability_of_target_using_explicit_sum(wire, target_binary_state="1"),
    ])
    np.testing.assert_allclose(
        es_m_probs, es_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_state,params,wire",
    [
        (np.random.randint(0, 2, size=n), mps.MatchgatePolarParams.random(), 0)
        for n in [2, ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_gate_circuit_probability_single_vs_target_random_cases(initial_binary_state, params, wire):
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperator(params, wires=[0, 1])
    ]
    device.apply(operations)
    es_probs = device.compute_probability_using_explicit_sum(wire)
    es_m_probs = np.asarray([
        device.compute_probability_of_target_using_explicit_sum(wire, target_binary_state=0),
        device.compute_probability_of_target_using_explicit_sum(wire, target_binary_state=1),
    ])
    np.testing.assert_allclose(
        es_m_probs, es_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
