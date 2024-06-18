import numpy as np
import pennylane as qml
import pytest
import psutil

from matchcake import MatchgateOperation, utils
from matchcake import matchgate_parameter_sets as mps
from . import devices_init
from .test_specific_circuit import specific_matchgate_circuit
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


def multiples_matchgate_circuit(params_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    if all_wires is None:
        all_wires = [0, 1]
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    for params in params_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        wire0 = np.random.choice(all_wires[:-1], size=1).item()
        wire1 = wire0 + 1
        MatchgateOperation(mg_params, wires=[wire0, wire1])
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "params_list,n_wires,prob_wires",
    [
        ([mps.MatchgatePolarParams.random(), mps.MatchgatePolarParams.random()], 4, 0)
    ]
    +
    [
        ([mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)], num_wires, 0)
        for num_wires in range(2, 6)
        for num_gates in [1, 2**num_wires]
    ]
    +
    [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], num_wires, 0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 2**num_wires]
    ]
)
def test_multiples_matchgate_probs_with_qbit_device(params_list, n_wires, prob_wires):
    nif_device, qubit_device = devices_init(wires=n_wires)
    
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


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "params_list,n_wires,prob_wires",
    [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], num_wires, 0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 5)
        for num_gates in [1, 2*num_wires]
    ]
)
def test_multiples_matchgate_probs_with_qbit_device_mp(params_list, n_wires, prob_wires):
    if psutil.cpu_count() < 2:
        pytest.skip("This test requires at least 2 CPUs.")
    n_workers = psutil.cpu_count()
    nif_device, qubit_device = devices_init(wires=n_wires, n_workers=n_workers)
    assert nif_device.n_workers == n_workers

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
