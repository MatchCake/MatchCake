from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils

np.random.seed(42)


def devices_init(*args, **kwargs) -> Tuple[NonInteractingFermionicDevice, qml.Device]:
    nif_device = NonInteractingFermionicDevice(wires=2)
    qubit_device = qml.device('default.qubit', wires=2, shots=kwargs.get("shots", 1))
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
    polar_params = mps.MatchgatePolarParams.parse_from_params(params)
    op = MatchgateOperator(polar_params, wires=[0, 1])
    qml.apply(op)
    return qml.probs(wires=0)


@pytest.mark.parametrize(
    "polar_params",
    # [
    #     np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # ]
    # +
    [
        np.random.rand(6)
        for _ in range(100)
    ]
)
def test_single_matchgate_expval_with_qbit_device(polar_params):
    nif_device, qubit_device = devices_init()
    
    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    
    nif_expectation_value = nif_qnode(polar_params)
    qubit_expectation_value = qubit_qnode(polar_params)
    check = np.allclose(nif_expectation_value, qubit_expectation_value)
    assert check, (f"The expectation value is not the correct one. "
                   f"Got {nif_expectation_value} instead of {qubit_expectation_value}")




