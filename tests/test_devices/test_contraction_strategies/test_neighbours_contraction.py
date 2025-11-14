import itertools

import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake import MatchgateOperation
from matchcake import matchgate_parameter_sets as mgp
from matchcake import utils
from matchcake.circuits import random_sptm_operations_generator
from matchcake.operations import SptmCompRxRx, SptmIdentity
from matchcake.utils import torch_utils

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import devices_init, init_nif_device
from ..test_single_line_matchgates_circuit import single_line_matchgates_circuit

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "operations,expected_new_operations",
    [
        (
            [MatchgateOperation(mgp.Identity, wires=[0, 1])],
            [MatchgateOperation(mgp.Identity, wires=[0, 1])],
        ),
    ],
)
def test_neighbours_contraction(operations, expected_new_operations):
    all_wires = set(wire for op in operations for wire in op.wires)
    nif_device_nh = init_nif_device(wires=len(all_wires), contraction_method="neighbours")
    new_operations = nif_device_nh.contraction_strategy(operations)

    assert len(new_operations) == len(expected_new_operations), "The number of operations is different."
    for new_op, expected_op in zip(new_operations, expected_new_operations):
        np.testing.assert_allclose(
            new_op.matrix(),
            expected_op.matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )


@pytest.mark.parametrize(
    "operations",
    [
        [MatchgateOperation(mgp.Identity, wires=[0, 1])],
        [MatchgateOperation(mgp.Identity, wires=[0, 1]) for _ in range(10)],
        [MatchgateOperation.random(batch_size=10, wires=[0, 1], seed=42)],
        [MatchgateOperation.random(batch_size=10, wires=[0, 1], seed=i) for i in range(2)],
    ],
)
def test_neighbours_contraction_device_one_line(operations):
    nif_device_nh = init_nif_device(wires=2, contraction_method="neighbours")
    nif_device = init_nif_device(wires=2, contraction_method=None)

    nif_device_nh.apply(operations)
    nif_device.apply(operations)

    np.testing.assert_allclose(
        nif_device.analytic_probability(),
        nif_device_nh.analytic_probability(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "operations",
    [
        [SptmIdentity(wires=[0, 1])],
        [SptmIdentity(wires=[0, 1]) for _ in range(10)],
        [SptmCompRxRx(np.random.random(2), wires=[0, 1])],
        [SptmCompRxRx(np.random.random(2), wires=[0, 1]) for _ in range(2)],
    ],
)
def test_neighbours_contraction_device_one_line_sptm(operations):
    nif_device_nh = init_nif_device(wires=2, contraction_method="neighbours")
    nif_device = init_nif_device(wires=2, contraction_method=None)

    nif_device_nh.apply(operations)
    nif_device.apply(operations)

    np.testing.assert_allclose(
        nif_device.analytic_probability(),
        nif_device_nh.analytic_probability(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,prob_wires",
    [
        ([mgp.MatchgatePolarParams(r0=1, r1=1) for _ in range(num_gates)], 0)
        for num_gates in 2 ** np.arange(1, 5)
    ]
    + [
        ([MatchgateOperation.random_params(seed=i) for i in range(num_gates)], 0)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_gates in 2 ** np.arange(1, 5)
    ],
)
def test_multiples_matchgate_probs_with_qbit_device_nh_contraction(params_list, prob_wires):
    nif_device, qubit_device = devices_init(wires=2, contraction_method="neighbours")

    nif_qnode = qml.QNode(single_line_matchgates_circuit, nif_device)
    qubit_qnode = qml.QNode(single_line_matchgates_circuit, qubit_device)

    initial_binary_state = np.array([0, 0])
    qubit_state = qubit_qnode(
        params_list,
        initial_binary_state,
        in_param_type=mgp.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
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


@pytest.mark.parametrize("x", [np.random.rand(4) for _ in range(N_RANDOM_TESTS_PER_CASE)])
def test_nh_contraction_torch_grad(x):
    n_qubits = 4
    x = torch.from_numpy(x).float()
    x_grad = x.detach().clone().requires_grad_(True)

    dev = mc.NonInteractingFermionicDevice(wires=n_qubits, contraction_method="neighbours")

    @qml.qnode(dev, interface="torch")
    def circuit(x):
        mc.operations.CompRyRy(x[0:2], wires=[0, 1])
        mc.operations.CompRyRy(x[2:4], wires=[2, 3])
        mc.operations.CompRyRy(x[0:2], wires=[0, 1])
        mc.operations.CompRyRy(x[2:4], wires=[2, 3])
        return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

    try:
        circuit(x_grad)
    except Exception as e:
        pytest.fail(f"Error during forward pass: {e}")

    np.testing.assert_allclose(
        torch_utils.to_numpy(circuit(x)),
        torch_utils.to_numpy(circuit(x_grad)),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="Forward pass with and without gradient computation are different.",
    )


@pytest.mark.parametrize(
    "circuit_gen, n_wires",
    [
        (
            random_sptm_operations_generator(n_ops=2 + i, wires=n_wires, batch_size=batch_size),
            n_wires,
        )
        for n_wires in range(2, 12)
        for i in range(N_RANDOM_TESTS_PER_CASE)
        for batch_size in [None, 16]
    ],
)
def test_nh_contraction_with_apply_generator(circuit_gen, n_wires):
    nif_device_nh = init_nif_device(wires=n_wires, contraction_method="neighbours")
    nif_device_none = init_nif_device(wires=n_wires, contraction_method=None)

    circuit_gen_nh, circuit_gen_none = itertools.tee(circuit_gen)

    nif_device_nh.execute_generator(circuit_gen_nh, apply=True, reset=True, cache_global_sptm=True)
    nh_sptm = nif_device_nh.apply_metadata["global_sptm"]

    nif_device_none.execute_generator(circuit_gen_none, apply=True, reset=True, cache_global_sptm=True)
    none_sptm = nif_device_none.apply_metadata["global_sptm"]

    np.testing.assert_allclose(
        nh_sptm,
        none_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="The SPTMs are different.",
    )


@pytest.mark.parametrize("n_wires", [n for n in range(2, 12)])
def test_nh_contraction_with_apply_generator_sptm_supp(n_wires):
    wires = np.arange(n_wires)

    def circuit_gen():
        for op in mc.operations.SptmFermionicSuperposition(wires=wires).decomposition():
            yield op
        return

    nif_device_nh = init_nif_device(wires=wires, contraction_method="neighbours")
    nif_device_none = init_nif_device(wires=wires, contraction_method=None)

    circuit_gen_nh, circuit_gen_none = itertools.tee(circuit_gen())

    nif_device_nh.execute_generator(circuit_gen_nh, apply=True, reset=True, cache_global_sptm=True)
    nh_sptm = nif_device_nh.apply_metadata["global_sptm"]

    nif_device_none.execute_generator(circuit_gen_none, apply=True, reset=True, cache_global_sptm=True)
    none_sptm = nif_device_none.apply_metadata["global_sptm"]

    np.testing.assert_allclose(
        nh_sptm,
        none_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="The SPTMs are different.",
    )


@pytest.mark.parametrize("n_wires", [n for n in range(2, 12)])
def test_nh_contraction_with_apply_generator_supp(n_wires):
    wires = np.arange(n_wires)

    def circuit_gen():
        for op in mc.operations.FermionicSuperposition(wires=wires).decomposition():
            yield op
        return

    nif_device_nh = init_nif_device(wires=wires, contraction_method="neighbours")
    nif_device_none = init_nif_device(wires=wires, contraction_method=None)

    circuit_gen_nh, circuit_gen_none = itertools.tee(circuit_gen())

    nif_device_nh.execute_generator(circuit_gen_nh, apply=True, reset=True, cache_global_sptm=True)
    nh_sptm = nif_device_nh.apply_metadata["global_sptm"]

    nif_device_none.execute_generator(circuit_gen_none, apply=True, reset=True, cache_global_sptm=True)
    none_sptm = nif_device_none.apply_metadata["global_sptm"]

    np.testing.assert_allclose(
        nh_sptm,
        none_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="The SPTMs are different.",
    )
