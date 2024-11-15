import numpy as np
import pennylane as qml
import pytest

import matchcake as mc
from matchcake import MatchgateOperation, utils
from matchcake import matchgate_parameter_sets as mps
from matchcake.operations import SptmRxRx, SptmIdentity
from .. import devices_init, init_nif_device
from ..test_single_line_matchgates_circuit import single_line_matchgates_circuit
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "operations,expected_new_operations",
    [
        (
            [MatchgateOperation(mps.Identity, wires=[0, 1])],
            [MatchgateOperation(mps.Identity, wires=[0, 1])],
        ),
    ]
)
def test_neighbours_contraction(operations, expected_new_operations):
    all_wires = set(wire for op in operations for wire in op.wires)
    nif_device_nh = init_nif_device(wires=len(all_wires), contraction_method="neighbours")
    new_operations = nif_device_nh.contraction_strategy(operations)

    assert len(new_operations) == len(expected_new_operations), "The number of operations is different."
    for new_op, expected_op in zip(new_operations, expected_new_operations):
        np.testing.assert_allclose(
            new_op.compute_matrix(), expected_op.compute_matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )


@pytest.mark.parametrize(
    "operations",
    [
        [MatchgateOperation(mps.Identity, wires=[0, 1])],
        [MatchgateOperation(mps.Identity, wires=[0, 1]) for _ in range(10)],
        [MatchgateOperation(mps.MatchgatePolarParams.random_batch_numpy(10), wires=[0, 1])],
        [MatchgateOperation(mps.MatchgatePolarParams.random_batch_numpy(10), wires=[0, 1]) for _ in range(2)],
    ]
)
def test_neighbours_contraction_device_one_line(operations):
    nif_device_nh = init_nif_device(wires=2, contraction_method="neighbours")
    nif_device = init_nif_device(wires=2, contraction_method=None)

    nif_device_nh.apply(operations)
    nif_device.apply(operations)

    np.testing.assert_allclose(
        nif_device.analytic_probability(), nif_device_nh.analytic_probability(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "operations",
    [
        [SptmIdentity(wires=[0, 1])],
        [SptmIdentity(wires=[0, 1]) for _ in range(10)],
        [SptmRxRx(np.random.random(2), wires=[0, 1])],
        [SptmRxRx(np.random.random(2), wires=[0, 1]) for _ in range(2)],
    ]
)
def test_neighbours_contraction_device_one_line_sptm(operations):
    nif_device_nh = init_nif_device(wires=2, contraction_method="neighbours")
    nif_device = init_nif_device(wires=2, contraction_method=None)

    nif_device_nh.apply(operations)
    nif_device.apply(operations)

    np.testing.assert_allclose(
        nif_device.analytic_probability(), nif_device_nh.analytic_probability(),
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


@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(4)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_nh_contraction_torch_grad(x):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    from matchcake.utils import torch_utils

    n_qubits = 4
    x = torch.from_numpy(x).float()
    x_grad = x.detach().clone().requires_grad_(True)

    dev = mc.NonInteractingFermionicDevice(
        wires=n_qubits,
        contraction_method="neighbours"
    )

    @qml.qnode(dev, interface="torch")
    def circuit(x):
        mc.operations.fRYY(x[0:2], wires=[0, 1])
        mc.operations.fRYY(x[2:4], wires=[2, 3])
        mc.operations.fRYY(x[0:2], wires=[0, 1])
        mc.operations.fRYY(x[2:4], wires=[2, 3])
        return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

    try:
        circuit(x_grad)
    except Exception as e:
        pytest.fail(f"Error during forward pass: {e}")

    np.testing.assert_allclose(
        torch_utils.to_numpy(circuit(x)), torch_utils.to_numpy(circuit(x_grad)),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="Forward pass with and without gradient computation are different."
    )
