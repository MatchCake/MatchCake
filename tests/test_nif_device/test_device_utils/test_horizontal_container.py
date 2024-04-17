import pytest
from matchcake.devices.device_utils import _HorizontalMatchgatesContainer, _SingleParticleTransitionMatrix
import matchcake as mc
from matchcake import matchgate_parameter_sets as mps
import numpy as np
import pennylane as qml

from .. import devices_init, init_nif_device
from ..test_specific_circuit import specific_matchgate_circuit
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "op",
    [
        mc.MatchgateOperation(mc.matchgate_parameter_sets.MatchgatePolarParams.random(10), wires=[wire, wire + 1])
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_horizontal_matchgates_container_contract_single_op(op):
    container = _HorizontalMatchgatesContainer()
    container.add(op)
    np.testing.assert_allclose(
        container.contract(),
        op.single_particle_transition_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "line_operations",
    [
        [
            mc.MatchgateOperation(mc.matchgate_parameter_sets.MatchgatePolarParams.random(10), wires=[wire, wire + 1])
            for _ in range(np.random.randint(1, 10))
        ]
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_horizontal_matchgates_container_contract_single_line(line_operations):
    container = _HorizontalMatchgatesContainer()
    assert container.contract() is None
    for op in line_operations:
        container.add(op)

    contract_ops = line_operations[0]
    for op in line_operations[1:]:
        contract_ops = contract_ops @ op

    wires = line_operations[0].wires

    np.testing.assert_allclose(
        container.op_container[wires].compute_matrix(),
        contract_ops.compute_matrix(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    target_t = contract_ops.single_particle_transition_matrix
    pred_t = container.contract()
    np.testing.assert_allclose(
        pred_t,
        target_t,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "column_operations",
    [
        [
            mc.MatchgateOperation(
                mc.matchgate_parameter_sets.MatchgatePolarParams.random(10),
                wires=[i, i + 1]
            )
            for i in range(0, np.random.randint(1, 20), 2)
        ]
        for _ in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_horizontal_matchgates_container_contract_single_column(column_operations):
    container = _HorizontalMatchgatesContainer()
    assert container.contract() is None
    new_operations = container.contract_operations(column_operations)
    assert len(new_operations) == len(column_operations)

    for op, new_op in zip(column_operations, new_operations):
        if isinstance(new_op, mc.MatchgateOperation):
            new_op = new_op.get_padded_single_particle_transition_matrix(op.wires)
        np.testing.assert_allclose(
            new_op,
            op.get_padded_single_particle_transition_matrix(op.wires),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )


@pytest.mark.parametrize(
    "operations",
    [
        [
            mc.MatchgateOperation(mc.matchgate_parameter_sets.MatchgatePolarParams.random(10), wires=[wire, wire + 1])
            for wire in range(0, n_lines, 2)
            for _ in range(n_columns)
        ]
        for n_lines, n_columns in np.random.randint(1, 10, (N_RANDOM_TESTS_PER_CASE, 2))
    ]
)
def test_horizontal_matchgates_container_contract_line_column(operations):
    container = _HorizontalMatchgatesContainer()
    assert container.contract() is None

    all_wires = set(wire for op in operations for wire in op.wires)
    contract_ops = operations[0].get_padded_single_particle_transition_matrix(all_wires)
    for op in operations[1:]:
        contract_ops = contract_ops @ op.get_padded_single_particle_transition_matrix(all_wires)

    pred_new_operations = container.contract_operations(operations)
    pred_contract_ops = pred_new_operations[0]
    if isinstance(pred_contract_ops, mc.MatchgateOperation):
        pred_contract_ops = pred_contract_ops.get_padded_single_particle_transition_matrix(all_wires)
    for op in pred_new_operations[1:]:
        if isinstance(op, mc.MatchgateOperation):
            op = op.get_padded_single_particle_transition_matrix(all_wires)
        pred_contract_ops = pred_contract_ops.pad(all_wires) @ op.pad(all_wires)

    np.testing.assert_allclose(
        pred_contract_ops,
        contract_ops,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params_list,n_wires",
    [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], num_wires)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 2 ** num_wires]
    ]
)
def test_multiples_matchgate_transition_matrix_with_nif_horizontal(params_list, n_wires):
    nif_device = init_nif_device(wires=n_wires, contraction_method=None)
    nif_device_contracted = init_nif_device(wires=n_wires, contraction_method="horizontal")

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    nif_qnode_contracted = qml.QNode(specific_matchgate_circuit, nif_device_contracted)

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1])
        for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="expval",
    )
    nif_qnode_contracted(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device_contracted.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="expval",
    )

    if not np.allclose(nif_device.transition_matrix, nif_device_contracted.transition_matrix):
        print(nif_device.transition_matrix)
        print(nif_device_contracted.transition_matrix)
    np.testing.assert_allclose(
        nif_device.transition_matrix,
        nif_device_contracted.transition_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
