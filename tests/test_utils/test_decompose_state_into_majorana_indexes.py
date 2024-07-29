import numpy as np
import pytest

from matchcake import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    ATOL_SHAPE_COMPARISON,
    RTOL_SHAPE_COMPARISON,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "state_idx",
    [
        np.random.randint(0, 2**n)
        for n in range(1, 10)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_binary_state_to_state(state_idx):
    binary_state = np.binary_repr(state_idx)
    state = utils.binary_state_to_state(binary_state)
    target_state = np.zeros(2**len(binary_state))
    target_state[state_idx] = 1
    np.testing.assert_allclose(
        state, target_state,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "binary_state,majorana_indexes",
    [
        ("10", [0]),
        ("01", [2]),
        ("11", [0, 2]),
        ("00", []),
        ("100", [0]),
        ("010", [2]),
        ("001", [4]),
        ("110", [0, 2]),
        ("101", [0, 4]),
        ("011", [2, 4]),
        ("111", [0, 2, 4]),
        ("000", []),
        ("1000", [0]),
        ("0100", [2]),
        ("0010", [4]),
        ("0001", [6]),
        ("1100", [0, 2]),
        ("1010", [0, 4]),
        ("1001", [0, 6]),
        ("0110", [2, 4]),
        ("0101", [2, 6]),
        ("0011", [4, 6]),
        ("1110", [0, 2, 4]),
        ("1101", [0, 2, 6]),
        ("1011", [0, 4, 6]),
        ("0111", [2, 4, 6]),
        ("1111", [0, 2, 4, 6]),
    ]
)
def test_decompose_binary_state_into_majorana_indexes_specific_cases(binary_state, majorana_indexes):
    pred_majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(binary_state)
    np.testing.assert_allclose(
        majorana_indexes, pred_majorana_indexes,
        atol=ATOL_SHAPE_COMPARISON,
        rtol=RTOL_SHAPE_COMPARISON,
    )


@pytest.mark.parametrize(
    "binary_state",
    [
        np.random.randint(0, 2, size=n)
        for n in range(1, 10)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_decompose_binary_state_into_majorana_indexes(binary_state):
    majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(binary_state)
    n = len(binary_state)
    state = utils.binary_state_to_state(binary_state).reshape(-1, 1)
    majoranas = [utils.get_majorana(i, n) for i in majorana_indexes]
    if len(majoranas) == 0:
        op = np.eye(2**n)
    else:
        op = utils.recursive_2in_operator(np.matmul, majoranas)
    zero_state = np.zeros((2**n, 1))
    zero_state[0] = 1
    predicted_state = op @ zero_state
    np.testing.assert_allclose(
        predicted_state, state,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
