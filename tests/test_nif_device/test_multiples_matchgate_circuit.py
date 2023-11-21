from functools import partial

import numpy as np
import pennylane as qml
import pytest

from msim import MatchgateOperator
from msim import matchgate_parameter_sets as mps
from . import devices_init
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


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
    +
    [
        ([mps.MatchgatePolarParams(
            r0=np.random.uniform(0.00, 0.01),
            r1=np.random.uniform(0.00, 0.01),
            theta0=np.random.uniform(-np.pi, np.pi) / 1,
            theta1=np.random.uniform(-np.pi, np.pi) / 1,
            theta2=np.random.uniform(-np.pi, np.pi) / 1,
            theta3=np.random.uniform(-np.pi, np.pi) / 1,
        ).to_numpy() for _ in range(num_gates)], num_wires)
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
