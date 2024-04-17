import pytest
from matchcake.devices.device_utils import _VHMatchgatesContainer, _SingleParticleTransitionMatrix
import matchcake as mc
from matchcake import matchgate_parameter_sets as mps
import numpy as np
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "new_op",
    [
        mc.MatchgateOperation(mc.matchgate_parameter_sets.MatchgatePolarParams.random(10), wires=[wire, wire + 1])
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_vh_matchgates_container_add(new_op):
    container = _VHMatchgatesContainer()
    container.add(new_op)
    assert len(container) == 1
    assert container.wires_set == set(new_op.wires.labels)
    assert container.op_container == {new_op.wires: new_op}


@pytest.mark.parametrize(
    "new_op0, new_op1, crossing_wires",
    [
        (
            mc.MatchgateOperation(mps.MatchgatePolarParams.random(10), wires=[0, 1]),
            mc.MatchgateOperation(mps.MatchgatePolarParams.random(10), wires=[0, 1]),
            False
        ),
        (
            mc.MatchgateOperation(mps.MatchgatePolarParams.random(10), wires=[0, 1]),
            mc.MatchgateOperation(mps.MatchgatePolarParams.random(10), wires=[1, 2]),
            True
        ),
        (
            mc.MatchgateOperation(mps.MatchgatePolarParams.random(10), wires=[0, 1]),
            mc.MatchgateOperation(mps.MatchgatePolarParams.random(10), wires=[2, 3]),
            False
        )
    ]
)
def test_vh_matchgates_container_try_add(new_op0, new_op1, crossing_wires):
    container = _VHMatchgatesContainer()
    container.add(new_op0)
    if crossing_wires:
        assert not container.try_add(new_op1)
        assert len(container) == 1
    elif new_op0.wires == new_op1.wires:
        assert container.try_add(new_op1)
        assert len(container) == 1
        assert container.wires_set == set(new_op0.wires.labels)
        np.testing.assert_allclose(
            container.op_container[new_op0.wires].compute_matrix(),
            (new_op0 @ new_op1).compute_matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
    else:
        assert container.try_add(new_op1)
        assert len(container) == 2
        assert container.wires_set == set(new_op0.wires.labels) | set(new_op1.wires.labels)
        np.testing.assert_allclose(
            container.op_container[new_op0.wires].compute_matrix(),
            new_op0.compute_matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
        np.testing.assert_allclose(
            container.op_container[new_op1.wires].compute_matrix(),
            new_op1.compute_matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )


@pytest.mark.parametrize(
    "op",
    [
        mc.MatchgateOperation(mc.matchgate_parameter_sets.MatchgatePolarParams.random(10), wires=[wire, wire + 1])
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_vh_matchgates_container_contract_single_op(op):
    container = _VHMatchgatesContainer()
    container.add(op)
    np.testing.assert_allclose(
        container.contract().compute_matrix(),
        op.compute_matrix(),
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
def test_vh_matchgates_container_contract_single_line(line_operations):
    container = _VHMatchgatesContainer()
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
                wires=[wire + i, wire + i + 1]
            )
            for i in range(np.random.randint(1, 10))
        ]
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_vh_matchgates_container_contract_single_line(column_operations):
    container = _VHMatchgatesContainer()
    assert container.contract() is None
    for op in column_operations:
        container.add(op)

    assert len(container) == len(column_operations)
    all_wires = set(wire for op in column_operations for wire in op.wires)

    contract_ops = column_operations[0].get_padded_single_particle_transition_matrix(all_wires)
    for op in column_operations[1:]:
        contract_ops = contract_ops @ op.get_padded_single_particle_transition_matrix(all_wires)

    np.testing.assert_allclose(
        container.contract(),
        contract_ops,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
