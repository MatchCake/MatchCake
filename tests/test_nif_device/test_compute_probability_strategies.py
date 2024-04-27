import numpy as np
import pennylane as qml
import pytest
import psutil

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake import matchgate_parameter_sets as mps
from matchcake import utils
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "initial_binary_state,params,wire",
    [
        (np.random.randint(0, 2, size=n), mps.MatchgatePolarParams.random(), 0)
        for n in [2, ]
        # for n in range(2, 10)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_gate_circuit_analytic_probability_lt_vs_es(initial_binary_state, params, wire):
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state), pfaffian_method="P")
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=[0, 1])
    ]
    device.apply(operations)
    lt_probs = device.compute_probability_using_lookup_table(wire)
    es_probs = device.compute_probability_using_explicit_sum(wire)
    np.testing.assert_allclose(
        lt_probs.squeeze(), es_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_string,params,wire,prob",
    [
        # Identity
        ("00", mps.Identity, 0, [1, 0]),
        ("01", mps.Identity, 0, [1, 0]),
        ("10", mps.Identity, 0, [0, 1]),
        ("11", mps.Identity, 0, [0, 1]),
        ("00", mps.Identity, 1, [1, 0]),
        ("01", mps.Identity, 1, [0, 1]),
        ("10", mps.Identity, 1, [1, 0]),
        ("11", mps.Identity, 1, [0, 1]),
        # fSWAP
        ("00", mps.fSWAP, 0, [1, 0]),
        ("01", mps.fSWAP, 0, [0, 1]),
        ("10", mps.fSWAP, 0, [1, 0]),
        ("11", mps.fSWAP, 0, [0, 1]),
        ("00", mps.fSWAP, 1, [1, 0]),
        ("01", mps.fSWAP, 1, [1, 0]),
        ("10", mps.fSWAP, 1, [0, 1]),
        ("11", mps.fSWAP, 1, [0, 1]),
        # Hell
        ("00", mps.HellParams, 0, [0.25, 0.75]),
        # ("01", mps.HellParams, 0, [0.25, 0.75]),
        # ("10", mps.HellParams, 0, [0.25, 0.75]),
        # ("11", mps.HellParams, 0, [0.25, 0.75]),
        ("00", mps.HellParams, 1, [0.25, 0.75]),
        # ("01", mps.HellParams, 1, [0.25, 0.75]),
        # ("10", mps.HellParams, 1, [0.25, 0.75]),
        # ("11", mps.HellParams, 1, [0.25, 0.75]),
    ]
)
def test_single_gate_circuit_analytic_probability_explicit_sum(initial_binary_string, params, wire, prob):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    prob = np.asarray(prob)
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=[0, 1])
    ]
    device.apply(operations)
    es_probs = device.compute_probability_using_explicit_sum(wire)
    np.testing.assert_allclose(
        prob.squeeze(), es_probs.squeeze(),
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
        for n in [2, ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_gate_circuit_probability_lt_vs_es(
        initial_binary_state, params, wires, target_binary_state
):
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=[0, 1])
    ]
    device.apply(operations)
    lt_probs = device.compute_probability_of_target_using_lookup_table(wires, target_binary_state)
    es_probs = device.compute_probability_of_target_using_explicit_sum(wires, target_binary_state)
    np.testing.assert_allclose(
        lt_probs, es_probs,
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
        for n in [2, ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_gate_circuit_probability_lt_vs_es_mp(
        initial_binary_state, params, wires, target_binary_state
):
    if psutil.cpu_count() < 2:
        pytest.skip("This test requires at least 2 CPUs.")
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state), n_workers=2)
    assert device.n_workers == 2
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=[0, 1])
    ]
    device.apply(operations)
    lt_probs = device.compute_probability_of_target_using_lookup_table(wires, target_binary_state)
    es_probs = device.compute_probability_of_target_using_explicit_sum(wires, target_binary_state)
    np.testing.assert_allclose(
        lt_probs, es_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
