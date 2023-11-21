from functools import partial

import numpy as np
import pennylane as qml
import pytest

from msim import MatchgateOperator, utils
from msim import matchgate_parameter_sets as mps
from . import devices_init
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


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
        MatchgateOperator(mg_params, wires=[wire0, wire1])
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@pytest.mark.parametrize(
    "params_list,n_wires,prob_wires",
    [
        ([mps.MatchgatePolarParams.random(), mps.MatchgatePolarParams.random()], 4, 0)
    ]
    # [
    #     ([mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)], num_wires, 0)
    #     for num_gates in range(1, 4)
    #     for num_wires in range(2, 6)
    # ]
    # +
    # [
    #     ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], num_wires, 0)
    #     for _ in range(N_RANDOM_TESTS_PER_CASE)
    #     for num_gates in range(1, 4)
    #     for num_wires in range(2, 6)
    # ]
)
def test_multiples_matchgate_probs_with_qbit_device(params_list, n_wires, prob_wires):
    nif_device, qubit_device = devices_init(wires=n_wires)

    nif_qnode = qml.QNode(multiples_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(multiples_matchgate_circuit, qubit_device)
    
    initial_binary_state = np.zeros(n_wires, dtype=int)
    qubit_state = qubit_qnode(
        params_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
    nif_probs = nif_qnode(
        params_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )
    
    check = np.allclose(nif_probs, qubit_probs, rtol=1.e-3, atol=1.e-3)
    assert check, f"The probs are not the correct one. Got {nif_probs} instead of {qubit_probs}"
