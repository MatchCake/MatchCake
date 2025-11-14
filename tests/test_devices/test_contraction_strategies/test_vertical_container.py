import numpy as np
import pennylane as qml
import pytest

import matchcake as mc
from matchcake import matchgate_parameter_sets as mgp
from matchcake.devices.contraction_strategies import get_contraction_strategy
from matchcake.utils.math import circuit_matmul, dagger, fermionic_operator_matmul

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import devices_init, init_nif_device
from .. import specific_matchgate_circuit

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "op",
    [
        mc.MatchgateOperation.random(
            batch_size=10,
            wires=[wire, wire + 1],
        )
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_vert_matchgates_container_contract_single_op(op):
    strategy = get_contraction_strategy("vertical")
    container = strategy.get_container()
    container.add(op)
    np.testing.assert_allclose(
        container.contract().matrix(),
        op.single_particle_transition_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "column_operations",
    [
        [
            mc.MatchgateOperation.random(
                batch_size=10,
                wires=[i, i + 1],
            )
            for i in range(0, np.random.randint(1, 20), 2)
        ]
        for _ in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_vert_matchgates_container_contract_single_column(column_operations):
    strategy = get_contraction_strategy("vertical")
    container = strategy.get_container()
    assert container.contract() is None
    for op in column_operations:
        container.add(op)

    assert len(container) == len(column_operations)
    all_wires = set(wire for op in column_operations for wire in op.wires)

    contract_ops = column_operations[0].get_padded_single_particle_transition_matrix(all_wires)
    for op in column_operations[1:]:
        contract_ops = contract_ops @ op.get_padded_single_particle_transition_matrix(all_wires)

    np.testing.assert_allclose(
        container.contract().matrix(),
        contract_ops.matrix(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "operations",
    [
        [
            mc.MatchgateOperation.random(
                batch_size=10,
                wires=[wire, wire + 1],
            )
            for wire in range(0, n_lines, 2)
            for _ in range(n_columns)
        ]
        for n_lines, n_columns in np.random.randint(1, 10, (N_RANDOM_TESTS_PER_CASE, 2))
    ],
)
def test_vert_matchgates_container_contract_line_column(operations):
    strategy = get_contraction_strategy("vertical")
    container = strategy.get_container()
    assert container.contract() is None

    all_wires = set(wire for op in operations for wire in op.wires)
    contract_ops = operations[0].to_sptm_operation().pad(all_wires)
    for op in operations[1:]:
        contract_ops = fermionic_operator_matmul(contract_ops, op.to_sptm_operation().pad(all_wires))

    pred_new_operations = container.contract_operations(operations)
    pred_contract_ops = pred_new_operations[0]
    if isinstance(pred_contract_ops, mc.MatchgateOperation):
        pred_contract_ops = pred_contract_ops.to_sptm_operation()
    for op in pred_new_operations[1:]:
        if isinstance(op, mc.MatchgateOperation):
            op = op.to_sptm_operation()
        pred_contract_ops = fermionic_operator_matmul(pred_contract_ops.pad(all_wires), op.pad(all_wires))

    pred_contract_matrix = pred_contract_ops.matrix()
    contract_matrix = contract_ops.matrix()
    # if not np.allclose(pred_contract_matrix, contract_matrix):
    #     test_vert_matchgates_container_contract_line_column(operations)
    #     print(pred_contract_matrix)
    #     print(contract_matrix)
    np.testing.assert_allclose(
        pred_contract_matrix,
        contract_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "operations",
    [
        [
            mc.MatchgateOperation.random(
                batch_size=10,
                wires=[wire, wire + 1],
            )
            for wire in range(0, n_lines, 2)
            for _ in range(n_columns)
        ]
        for n_lines, n_columns in np.random.randint(1, 10, (N_RANDOM_TESTS_PER_CASE, 2))
    ],
)
def test_vert_matchgates_container_contract_line_column_probs(operations):
    all_wires = set(wire for op in operations for wire in op.wires)
    nif_device = init_nif_device(wires=all_wires, contraction_method=None)
    nif_device_contracted = init_nif_device(wires=all_wires, contraction_method="vertical")

    nif_device.apply(operations)
    nif_device_contracted.apply(operations)

    nif_probs = nif_device.analytic_probability()
    nif_contract_probs = nif_device_contracted.analytic_probability()

    np.testing.assert_allclose(
        nif_contract_probs,
        nif_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,n_wires",
    [
        (
            [mc.MatchgateOperation.random_params(seed=i) for i in range(num_gates)],
            num_wires,
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
    ],
)
def test_multiples_matchgate_probs_with_nif_vertical(params_list, n_wires):
    all_wires = np.arange(n_wires)
    nif_device = init_nif_device(wires=all_wires, contraction_method=None)
    nif_device_contracted = init_nif_device(wires=all_wires, contraction_method="vertical")

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    nif_qnode_contracted = qml.QNode(specific_matchgate_circuit, nif_device_contracted)

    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    nif_probs = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mgp.MatchgatePolarParams,
        out_op="probs",
    )
    nif_contract_probs = nif_qnode_contracted(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device_contracted.wires,
        in_param_type=mgp.MatchgatePolarParams,
        out_op="probs",
    )

    np.testing.assert_allclose(
        nif_probs.sum(),
        1.0,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="The sum of the probabilities should be 1",
    )

    np.testing.assert_allclose(
        nif_contract_probs.squeeze(),
        nif_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,n_wires",
    [
        (
            [mc.MatchgateOperation.random_params(seed=i) for i in range(num_gates)],
            num_wires,
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
    ],
)
def test_multiples_matchgate_probs_with_qubits_device_vertical(params_list, n_wires):
    all_wires = np.arange(n_wires)
    qubit_device = qml.device("default.qubit", wires=len(all_wires), shots=None)
    nif_device_contracted = init_nif_device(wires=all_wires, contraction_method="vertical")

    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)
    nif_qnode_contracted = qml.QNode(specific_matchgate_circuit, nif_device_contracted)

    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    nif_probs = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mgp.MatchgatePolarParams,
        out_op="probs",
    )
    nif_contract_probs = nif_qnode_contracted(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device_contracted.wires,
        in_param_type=mgp.MatchgatePolarParams,
        out_op="probs",
    )

    np.testing.assert_allclose(
        nif_probs.sum(),
        1.0,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg="The sum of the probabilities should be 1",
    )

    np.testing.assert_allclose(
        nif_contract_probs.squeeze(),
        nif_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
