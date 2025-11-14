import numpy as np
import pytest

import matchcake as mc
from matchcake import matchgate_parameter_sets as mgp
from matchcake.devices.contraction_strategies import get_contraction_strategy
from matchcake.operations import SptmCompRxRx

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "new_op",
    [
        mc.MatchgateOperation.random(
            batch_size=10,
            wires=[wire, wire + 1],
        )
        for wire in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_vh_matchgates_container_add(new_op):
    strategy = get_contraction_strategy("neighbours")
    container = strategy.get_container()
    container.add(new_op)
    assert len(container) == 1
    assert container.wires_set == set(new_op.wires.labels)
    assert container.op_container == {new_op.wires: new_op}


@pytest.mark.parametrize(
    "new_op0, new_op1, crossing_wires",
    [
        (
            mc.MatchgateOperation.random(batch_size=10, wires=[0, 1]),
            mc.MatchgateOperation.random(batch_size=10, wires=[0, 1]),
            False,
        ),
        (
            mc.MatchgateOperation.random(batch_size=10, wires=[0, 1]),
            mc.MatchgateOperation.random(batch_size=10, wires=[1, 2]),
            True,
        ),
        (
            mc.MatchgateOperation.random(batch_size=10, wires=[0, 1]),
            mc.MatchgateOperation.random(batch_size=10, wires=[2, 3]),
            False,
        ),
    ],
)
def test_vh_matchgates_container_try_add(new_op0, new_op1, crossing_wires):
    strategy = get_contraction_strategy("neighbours")
    container = strategy.get_container()
    container.add(new_op0)
    if crossing_wires:
        assert not container.try_add(new_op1)
        assert len(container) == 1
    elif new_op0.wires == new_op1.wires:
        assert container.try_add(new_op1)
        assert len(container) == 1
        assert container.wires_set == set(new_op0.wires.labels)
        np.testing.assert_allclose(
            container.op_container[new_op0.wires].matrix(),
            (new_op0 @ new_op1).matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
    else:
        assert container.try_add(new_op1)
        assert len(container) == 2
        assert container.wires_set == set(new_op0.wires.labels) | set(new_op1.wires.labels)
        np.testing.assert_allclose(
            container.op_container[new_op0.wires].matrix(),
            new_op0.matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
        np.testing.assert_allclose(
            container.op_container[new_op1.wires].matrix(),
            new_op1.matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )


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
def test_vh_matchgates_container_contract_single_op(op):
    strategy = get_contraction_strategy("neighbours")
    container = strategy.get_container()
    container.add(op)
    np.testing.assert_allclose(
        container.contract().matrix(),
        op.single_particle_transition_matrix,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "line_operations",
    [
        [mc.MatchgateOperation.random(batch_size=1, wires=[0, 1]) for _ in range(n_gates)]
        for n_gates in np.arange(1, N_RANDOM_TESTS_PER_CASE + 1)
    ],
)
def test_vh_matchgates_container_contract_single_line(line_operations):
    strategy = get_contraction_strategy("neighbours")
    container = strategy.get_container()
    assert container.contract() is None
    for op in line_operations:
        container.add(op)

    contract_ops = line_operations[0]
    for op in line_operations[1:]:
        contract_ops = op @ contract_ops

    wires = line_operations[0].wires

    np.testing.assert_allclose(
        container.op_container[wires].matrix(),
        contract_ops.matrix(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    target_t = contract_ops.single_particle_transition_matrix
    pred_t = container.contract().matrix()
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
            mc.MatchgateOperation.random(
                batch_size=10,
                wires=[i, i + 1],
            )
            for i in range(0, np.random.randint(1, 20), 2)
        ]
        for _ in np.random.randint(0, 2, N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_vh_matchgates_container_contract_single_column(column_operations):
    strategy = get_contraction_strategy("neighbours")
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
    "new_op0, new_op1, crossing_wires",
    [
        (
            SptmCompRxRx(np.random.random(2), wires=[0, 1]),
            SptmCompRxRx(np.random.random(2), wires=[0, 1]),
            False,
        ),
        (
            SptmCompRxRx(np.random.random(2), wires=[0, 1]),
            mc.MatchgateOperation.random(batch_size=10, wires=[1, 2]),
            True,
        ),
        (
            SptmCompRxRx(np.random.random(2), wires=[0, 1]),
            SptmCompRxRx(np.random.random(2), wires=[1, 2]),
            True,
        ),
        (
            SptmCompRxRx(np.random.random(2), wires=[0, 1]),
            mc.MatchgateOperation.random(batch_size=10, wires=[2, 3]),
            False,
        ),
        (
            SptmCompRxRx(np.random.random(2), wires=[0, 1]),
            SptmCompRxRx(np.random.random(2), wires=[2, 3]),
            False,
        ),
    ],
)
def test_vh_matchgates_container_try_add_sptm(new_op0, new_op1, crossing_wires):
    strategy = get_contraction_strategy("neighbours")
    container = strategy.get_container()
    container.add(new_op0)
    if crossing_wires:
        assert not container.try_add(new_op1)
        assert len(container) == 1
    elif new_op0.wires == new_op1.wires:
        assert container.try_add(new_op1)
        assert len(container) == 1
        assert container.wires_set == set(new_op0.wires.labels)
        np.testing.assert_allclose(
            container.op_container[new_op0.wires].matrix(),
            (new_op0 @ new_op1).matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
    else:
        assert container.try_add(new_op1)
        assert len(container) == 2
        assert container.wires_set == set(new_op0.wires.labels) | set(new_op1.wires.labels)
        np.testing.assert_allclose(
            container.op_container[new_op0.wires].matrix(),
            new_op0.matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
        np.testing.assert_allclose(
            container.op_container[new_op1.wires].matrix(),
            new_op1.matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
