import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation, utils
from matchcake import matchgate_parameter_sets as mps
from . import devices_init, init_nif_device
from .test_single_line_matchgates_circuit import single_line_matchgates_circuit
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "operations,expected_new_operations",
    [
        (
            [MatchgateOperation(mps.Identity, wires=[0, 1])],
            [MatchgateOperation(mps.Identity, wires=[0, 1])],
        )
    ]
)
def test_neighbours_contraction(operations, expected_new_operations):
    all_wires = set(wire for op in operations for wire in op.wires)
    nif_device_nh = init_nif_device(wires=len(all_wires), contraction_method="neighbours")
    new_operations = nif_device_nh.do_neighbours_contraction(operations)

    assert len(new_operations) == len(expected_new_operations), "The number of operations is different."
    for new_op, expected_op in zip(new_operations, expected_new_operations):
        np.testing.assert_allclose(
            new_op.compute_matrix(), expected_op.compute_matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )


def test_neighbours_contraction_device(operations):
    nif_device_nh = init_nif_device(wires=2, contraction_method="neighbours")
    nif_device = init_nif_device(wires=2, contraction_method=None)

    nif_device_nh.apply(operations)
    nif_device.apply(operations)

    np.testing.assert_allclose(
        nif_device.transition_matrix, nif_device_nh.transition_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,prob_wires",
    [
        ([mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)], 0)
        for num_gates in 2 ** np.arange(1, 5)
    ]
    +
    [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], 0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_gates in 2 ** np.arange(1, 5)
    ]
)
def test_multiples_matchgate_probs_with_qbit_device_nh_contraction(params_list, prob_wires):
    nif_device, qubit_device = devices_init(wires=2, contraction_method="neighbours")

    nif_qnode = qml.QNode(single_line_matchgates_circuit, nif_device)
    qubit_qnode = qml.QNode(single_line_matchgates_circuit, qubit_device)

    initial_binary_state = np.array([0, 0])
    qubit_state = qubit_qnode(
        params_list,
        initial_binary_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
    nif_probs = nif_qnode(
        params_list,
        initial_binary_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )

    np.testing.assert_allclose(
        nif_probs.squeeze(), qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )