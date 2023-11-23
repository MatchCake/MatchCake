import numpy as np
import pytest

from msim import utils
from msim.lookup_table import NonInteractingFermionicLookupTable
from tests.configs import N_RANDOM_TESTS_PER_CASE, ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "transition_matrix, binary_state",
    [
        (np.random.rand(len(s), 2 * len(s)), s)
        for s in ["00", "01", "10", "11"]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_observable_form(transition_matrix, binary_state):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    binary_state = utils.binary_string_to_vector(binary_state)
    hamming_weight = np.sum(binary_state, dtype=int)
    state = utils.binary_state_to_state(binary_state)
    
    obs = lookup_table.get_observable(0, state)
    np.testing.assert_allclose(
        obs.shape, (2 * hamming_weight + 2, 2 * hamming_weight + 2),
        err_msg="The observable has the wrong shape."
    )
    np.testing.assert_allclose(
        obs + obs.T, np.zeros_like(obs),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The observable is not symmetric."
    )
    np.testing.assert_allclose(
        np.diagonal(obs), np.zeros(obs.shape[0]),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The diagonal of the observable is not zero."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item00(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    TBT = transition_matrix @ B @ transition_matrix.T
    np.testing.assert_allclose(
        lookup_table[0, 0], TBT,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (0, 0) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item01(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = transition_matrix @ B @ np.conjugate(transition_matrix.T)
    np.testing.assert_allclose(
        lookup_table[0, 1], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (0, 1) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item02(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = transition_matrix @ B
    np.testing.assert_allclose(
        lookup_table[0, 2], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (0, 2) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item10(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B @ transition_matrix.T
    np.testing.assert_allclose(
        lookup_table[1, 0], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (1, 0) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item11(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B @ np.conjugate(transition_matrix.T)
    np.testing.assert_allclose(
        lookup_table[1, 1], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (1, 1) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item12(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B
    np.testing.assert_allclose(
        lookup_table[1, 2], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (1, 2) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item20(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = B @ transition_matrix.T
    np.testing.assert_allclose(
        lookup_table[2, 0], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (2, 0) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item21(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = B @ np.conjugate(transition_matrix.T)
    np.testing.assert_allclose(
        lookup_table[2, 1], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (2, 1) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2 * n)
        for n in range(2, 5)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item22(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    item = np.eye(transition_matrix.shape[-1])
    np.testing.assert_allclose(
        lookup_table[2, 2], item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (2, 2) is not correct."
    )


@pytest.mark.parametrize(
    "transition_matrix,binary_state,k,observable",
    [
        #
        (
                0.5 * np.array(
                    [
                        [1, 1j, 0, 0],
                        [0, 0, 1, 1j]
                    ]
                ),
                "00", 0,  # binary_state, k
                np.array(
                    [
                        [0, 0],
                        [0, 0],
                    ]
                )
        ),
        #
        (
                0.5 * np.array(
                    [
                        [0, 0, 1, 1j],
                        [1, 1j, 0, 0]
                    ]
                ),
                "01", 0,  # binary_state, k
                np.array(
                    [
                        [0, 1, 0, 1],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [-1, 0, -1, 0],
                    ]
                )
        ),
        #
    ]
)
def test_lookup_table_get_observable(transition_matrix, binary_state, k, observable):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    obs = lookup_table.get_observable(k, utils.binary_state_to_state(binary_state))
    np.testing.assert_allclose(
        obs, observable,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
