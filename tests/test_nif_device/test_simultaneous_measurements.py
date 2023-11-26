import numpy as np
import pennylane as qml
import pytest

from msim import MatchgateOperator, NonInteractingFermionicDevice
from msim import matchgate_parameter_sets as mps
from msim import utils
from . import devices_init
from .test_specific_circuit import specific_matchgate_circuit
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
        device.compute_probability_of_target_using_explicit_sum(wire, target_binary_state="0"),
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
    

@pytest.mark.parametrize(
    "initial_binary_string,params,wires,target_binary_state,prob",
    [
        ("01", mps.fSWAP, [0, 1], "10", 1),
    ]
)
def test_single_gate_circuit_probability_target_state_specific_cases(
        initial_binary_string, params, wires, target_binary_state, prob
):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    device = NonInteractingFermionicDevice(wires=wires)
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperator(params, wires=wires)
    ]
    device.apply(operations)
    es_m_prob = device.compute_probability_of_target_using_explicit_sum(wires, target_binary_state=target_binary_state)
    np.testing.assert_allclose(
        es_m_prob, prob,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "params_list,n_wires,prob_wires",
    [
        (
                [mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)],
                num_wires, np.random.choice(num_wires, replace=False, size=n_probs)
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in [2, 3]
        for num_gates in [1, 2*num_wires]
        for n_probs in range(1, num_wires+1)
    ]
)
def test_multiples_matchgate_probs_with_qbit_device_explicit_sum(params_list, n_wires, prob_wires):
    nif_device, qubit_device = devices_init(wires=n_wires, prob_strategy="explicit_sum")

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1])
        for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    qubit_state = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
    nif_probs = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )
    np.testing.assert_allclose(
        nif_probs, qubit_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,n_wires,prob_wires",
    [
        (
                [mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)],
                num_wires, np.random.choice(num_wires, replace=False, size=n_probs),
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in [2, 3, 4]
        for num_gates in [1, 2*num_wires]
        for n_probs in range(1, num_wires+1)
    ]
)
def test_multiples_matchgate_probs_with_qbit_device_lookup_table(params_list, n_wires, prob_wires):
    nif_device, qubit_device = devices_init(wires=n_wires, prob_strategy="lookup_table")

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1])
        for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    qubit_state = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
    nif_probs = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )
    np.testing.assert_allclose(
        nif_probs, qubit_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
