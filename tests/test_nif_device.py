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
    "params,target_expectation_value",
    [
        (mps.MatchgateStandardParams(a=1, w=1, z=1, d=1), np.array([1.0, 0.0])),
        (mps.MatchgateStandardParams(a=-1, w=-1, z=-1, d=-1), np.array([1.0, 0.0])),
        (mps.MatchgatePolarParams(r0=0, r1=1, theta1=0), np.array([0.0, 1.0])),
    ]
)
def test_single_gate_circuit_analytic_probability(params, target_expectation_value):
    device = NonInteractingFermionicDevice(wires=2)
    op = MatchgateOperator(params, wires=[0, 1])
    device.apply(op)
    expectation_value = device.analytic_probability(0)
    check = np.allclose(expectation_value, target_expectation_value)
    assert check, (f"The expectation value is not the correct one. "
                   f"Got {expectation_value} instead of {target_expectation_value}")


def single_matchgate_circuit(params):
    MatchgateOperator(params, wires=[0, 1])
    return qml.probs(wires=0)


@pytest.mark.parametrize(
    "params",
    [
        mps.MatchgateComposedHamiltonianParams()
    ]
    +
    [
        mps.MatchgatePolarParams(r0=1, r1=1),
        mps.MatchgatePolarParams(r0=0, r1=1, theta1=0)
    ]
    +
    [
        mps.MatchgatePolarParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_single_matchgate_probs_with_qbit_device(params):
    nif_device, qubit_device = devices_init()
    
    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    
    nif_probs = nif_qnode(mps.MatchgatePolarParams.parse_from_any(params).to_numpy())
    qubit_probs = qubit_qnode(mps.MatchgatePolarParams.parse_from_any(params).to_numpy())
    same_argmax = np.argmax(nif_probs) == np.argmax(qubit_probs)
    assert same_argmax, (f"The argmax is not the correct one. "
                         f"Got {np.argmax(nif_probs)} instead of {np.argmax(qubit_probs)}")
    check = np.allclose(nif_probs, qubit_probs, rtol=1.e-1, atol=1.e-1)
    assert check, f"The probs are not the correct one. Got {nif_probs} instead of {qubit_probs}"


def multiples_matchgate_circuit(params_list, all_wires=None):
    if all_wires is None:
        all_wires = [0, 1]
    all_wires = np.sort(np.asarray(all_wires))
    for params in params_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        wire0 = np.random.choice(all_wires[:-1], size=1).item()
        wire1 = wire0 + 1
        MatchgateOperator(mg_params, wires=[wire0, wire1])
    return qml.probs(wires=0)


@pytest.mark.parametrize(
    "params_list,n_wires",
    [
        ([mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)], num_wires)
        for num_gates in range(1, 3)
        for num_wires in range(2, 6)
    ]
    +
    [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], num_wires)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_gates in range(1, 3)
        for num_wires in range(2, 6)
    ]
)
def test_multiples_matchgate_probs_with_qbit_device(params_list, n_wires):
    nif_device, qubit_device = devices_init(wires=n_wires)

    multiples_matchgate_circuit_func = partial(multiples_matchgate_circuit, all_wires=list(range(n_wires)))
    nif_qnode = qml.QNode(multiples_matchgate_circuit_func, nif_device)
    qubit_qnode = qml.QNode(multiples_matchgate_circuit_func, qubit_device)

    nif_probs = nif_qnode(params_list)
    qubit_probs = qubit_qnode(params_list)
    same_argmax = np.argmax(nif_probs) == np.argmax(qubit_probs)
    assert same_argmax, (f"The argmax is not the correct one. "
                         f"Got {np.argmax(nif_probs)} instead of {np.argmax(qubit_probs)}")
    check = np.allclose(nif_probs, qubit_probs, rtol=1.e-1, atol=1.e-1)
    assert check, f"The probs are not the correct one. Got {nif_probs} instead of {qubit_probs}"
