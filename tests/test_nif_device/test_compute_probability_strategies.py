from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils
from functools import partial

from msim.utils import PAULI_Z, PAULI_X
from . import devices_init
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


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
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperator(params, wires=[0, 1])
    ]
    device.apply(operations)
    lt_probs = device.compute_probability_using_lookup_table(wire)
    es_probs = device.compute_probability_using_explicit_sum(wire)
    check = np.allclose(lt_probs, es_probs)
    assert check, (f"The probabilities are not the same. "
                   f"Got {lt_probs=} and {es_probs=}.")


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
    ]
)
def test_single_gate_circuit_analytic_probability_explicit_sum(initial_binary_string, params, wire, prob):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    prob = np.asarray(prob)
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperator(params, wires=[0, 1])
    ]
    device.apply(operations)
    es_probs = device.compute_probability_using_explicit_sum(wire)
    check = np.allclose(prob, es_probs)
    assert check, f"The probabilities are not the same. Got {es_probs=} and {prob=}. {initial_binary_string=}"
