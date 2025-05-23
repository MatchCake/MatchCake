import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation, utils
from matchcake import matchgate_parameter_sets as mps
from . import devices_init
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


def single_line_matchgates_circuit(params_list, initial_state=None, **kwargs):
    all_wires = [0, 1]
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(2 ** len(all_wires))
    qml.BasisState(initial_state, wires=all_wires)
    for params in params_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        MatchgateOperation(mg_params, wires=all_wires)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@pytest.mark.parametrize(
    "params_list,prob_wires",
    [
        ([mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)], 0)
        for num_gates in 10 * np.arange(1, 5)
    ]
    + [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], 0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_gates in 10 * np.arange(1, 5)
    ],
)
def test_multiples_matchgate_probs_with_qbit_device(params_list, prob_wires):
    nif_device, qubit_device = devices_init(wires=2)

    nif_qnode = qml.QNode(single_line_matchgates_circuit, nif_device)
    qubit_qnode = qml.QNode(single_line_matchgates_circuit, qubit_device)

    initial_binary_state = np.array([0, 0])
    qubit_probs = qubit_qnode(
        params_list,
        initial_binary_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )
    nif_probs = nif_qnode(
        params_list,
        initial_binary_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )

    np.testing.assert_allclose(
        nif_probs.squeeze(),
        qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
