import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake import matchgate_parameter_sets as mps
from matchcake import utils
from . import devices_init
from .test_specific_circuit import specific_matchgate_circuit
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)
    

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
    device = NonInteractingFermionicDevice(wires=wires, prob_strategy="ExplicitSum")
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperation(params, wires=wires)
    ]
    device.apply(operations)
    es_m_prob = device.get_state_probability(target_binary_state=target_binary_state, wires=wires)
    np.testing.assert_allclose(
        es_m_prob.squeeze(), prob,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@get_slow_test_mark()
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
        for num_gates in [2*num_wires]
        for n_probs in range(1, num_wires+1)
    ]
)
def test_multiples_matchgate_probs_with_qbit_device_explicit_sum(params_list, n_wires, prob_wires):
    nif_device, qubit_device = devices_init(wires=n_wires, prob_strategy="ExplicitSum")

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
        nif_probs.squeeze(), qubit_probs.squeeze(),
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
    all_wires = np.arange(n_wires)
    nif_device, qubit_device = devices_init(wires=all_wires, prob_strategy="LookupTable")

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1])
        for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    qubit_state = qml.expval(qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    ))
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
        nif_probs.squeeze(), qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
