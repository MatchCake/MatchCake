from functools import partial

import numpy as np
import pennylane as qml
import pytest

from msim import MatchgateOperator, utils
from msim import matchgate_parameter_sets as mps
from . import devices_init

np.random.seed(42)


def specific_matchgate_circuit(params_wires_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    if all_wires is None:
        all_wires = set(sum([list(wires) for _, wires in params_wires_list], []))
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    for params, wires in params_wires_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        MatchgateOperator(mg_params, wires=wires)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@pytest.mark.parametrize(
    "initial_binary_string,params_wires_list,prob_wires",
    [
        ("00", [(mps.fSWAP, [0, 1])], 0),
        ("01", [(mps.fSWAP, [0, 1])], 0),
        ("10", [(mps.fSWAP, [0, 1])], 0),
        ("11", [(mps.fSWAP, [0, 1])], 0),
        ("00", [(mps.HellParams, [0, 1])], 0),
        ("01", [(mps.HellParams, [0, 1])], 0),
        ("10", [(mps.HellParams, [0, 1])], 0),
        ("11", [(mps.HellParams, [0, 1])], 0),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3])], 0),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3]), (mps.fSWAP, [4, 5])], 0),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.fSWAP, [2, 3])], 0),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3])], 0),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3]), (mps.fSWAP, [4, 5])], 0),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.HellParams, [2, 3])], 0),
    ]
)
def test_multiples_matchgate_probs_with_qbit_device(initial_binary_string, params_wires_list, prob_wires):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

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

    check = np.allclose(nif_probs, qubit_probs, rtol=1.e-3, atol=1.e-3)
    assert check, f"The probs are not the correct one. Got {nif_probs} instead of {qubit_probs}"
