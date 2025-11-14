import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation
from matchcake import matchgate_parameter_sets as mgp
from matchcake import utils

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import devices_init, single_line_matchgates_circuit

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "params_list,prob_wires",
    [
        ([mgp.MatchgatePolarParams(r0=1, r1=1) for _ in range(num_gates)], 0)
        for num_gates in 10 * np.arange(1, 5)
    ]
    + [
        ([MatchgateOperation.random_params(seed=i) for i in range(num_gates)], 0)
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
        in_param_type=mgp.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )
    nif_probs = nif_qnode(
        params_list,
        initial_binary_state,
        in_param_type=mgp.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )

    np.testing.assert_allclose(
        nif_probs.squeeze(),
        qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
