from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils
from functools import partial
from .configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.Device]:
    nif_device = NonInteractingFermionicDevice(wires=kwargs.get("wires", 2))
    qubit_device = qml.device('default.qubit', wires=kwargs.get("wires", 2), shots=kwargs.get("shots", 1))
    qubit_device.operations.add(MatchgateOperator)
    return nif_device, qubit_device


@pytest.mark.parametrize(
    "gate,target_expectation_value",
    [
        (np.eye(4), 0.0),
    ]
)
def test_single_gate_circuit_expectation_value(gate, target_expectation_value):
    device = NonInteractingFermionicDevice(wires=2)
    mg_params = Matchgate.from_matrix(gate).polar_params
    op = MatchgateOperator(mg_params, wires=[0, 1])
    device.apply(op)
    expectation_value = device.analytic_probability(0)
    check = np.isclose(expectation_value, target_expectation_value)
    assert check, (f"The expectation value is not the correct one. "
                   f"Got {expectation_value} instead of {target_expectation_value}")


def single_matchgate_circuit(params):
    h_params = mps.MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
    op = MatchgateOperator(h_params, wires=[0, 1])
    qml.apply(op)
    return qml.probs(wires=0)


@pytest.mark.parametrize(
    "h_params",
    [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ]
    +
    [
        np.random.rand(6)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_matchgate_probs_with_qbit_device(h_params):
    nif_device, qubit_device = devices_init()
    
    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    
    nif_probs = nif_qnode(h_params)
    qubit_probs = qubit_qnode(h_params)
    check = np.allclose(nif_probs, qubit_probs)
    assert check, f"The probs are not the correct one. Got {nif_probs} instead of {qubit_probs}"


def multiples_matchgate_circuit(h_params_list, all_wires=None):
    if all_wires is None:
        all_wires = [0, 1]
    all_wires = np.sort(np.asarray(all_wires))
    for params in h_params_list:
        h_params = mps.MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
        wire0 = np.random.choice(all_wires[:-1], size=1).item()
        wire1 = wire0 + 1
        op = MatchgateOperator(h_params, wires=[wire0, wire1])
        qml.apply(op)
    return qml.probs(wires=0)


@pytest.mark.parametrize(
    "h_params_list,n_wires",
    [
        ([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(num_gates)], num_wires)
        for num_gates in range(1, 3)
        for num_wires in range(2, 6)
    ]
    +
    [
        ([np.random.rand(6) for _ in range(num_gates)], num_wires)
        for _ in range(10)
        for num_gates in range(1, 3)
        for num_wires in range(2, 6)
    ]
)
def test_multiples_matchgate_probs_with_qbit_device(h_params_list, n_wires):
    nif_device, qubit_device = devices_init(wires=n_wires)

    multiples_matchgate_circuit_func = partial(multiples_matchgate_circuit, all_wires=list(range(n_wires)))
    nif_qnode = qml.QNode(multiples_matchgate_circuit_func, nif_device)
    qubit_qnode = qml.QNode(multiples_matchgate_circuit_func, qubit_device)

    nif_probs = nif_qnode(h_params_list)
    qubit_probs = qubit_qnode(h_params_list)
    check = np.allclose(nif_probs, qubit_probs)
    assert check, f"The probs are not the correct one. Got {nif_probs} instead of {qubit_probs}"
