import pytest
import numpy as np
from msim import utils
from msim.lookup_table import NonInteractingFermionicLookupTable
from tests.configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.mark.parametrize(
    "transition_matrix, binary_state",
    [
        (np.random.rand(len(s), 2*len(s)), s)
        for s in ["00", "01", "10", "11"]
        # for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_observable_form(transition_matrix, binary_state):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    state = utils.binary_state_to_state(binary_state)
    hamming_weight = utils.get_hamming_weight(state)
    obs = lookup_table.get_observable(0, state)
    assert obs.shape == (2 * hamming_weight + 2, 2 * hamming_weight + 2), "The observable has the wrong shape."
    assert np.allclose(obs+obs.T, np.zeros_like(obs)), "The observable is not anti-symmetric."
    assert np.allclose(np.diagonal(obs), np.zeros(obs.shape[0])), "The diagonal of the observable is not zero."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item00(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    TBT = transition_matrix @ B @ transition_matrix.T
    assert np.allclose(lookup_table[0, 0], TBT), "The item (0, 0) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item01(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = transition_matrix @ B @ np.conjugate(transition_matrix.T)
    assert np.allclose(lookup_table[0, 1], item), "The item (0, 1) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item02(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = transition_matrix @ B
    assert np.allclose(lookup_table[0, 2], item), "The item (0, 2) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item10(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B @ transition_matrix.T
    assert np.allclose(lookup_table[1, 0], item), "The item (1, 0) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item11(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B @ np.conjugate(transition_matrix.T)
    assert np.allclose(lookup_table[1, 1], item), "The item (1, 1) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item12(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B
    assert np.allclose(lookup_table[1, 2], item), "The item (1, 2) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item20(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = B @ transition_matrix.T
    assert np.allclose(lookup_table[2, 0], item), "The item (2, 0) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item21(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = B @ np.conjugate(transition_matrix.T)
    assert np.allclose(lookup_table[2, 1], item), "The item (2, 1) is not correct."


@pytest.mark.parametrize(
    "transition_matrix",
    [
        np.random.rand(n, 2*n)
        for n in [2, ]
        for h in [0, 1]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ]
)
def test_lookup_table_item22(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    item = np.eye(transition_matrix.shape[-1])
    assert np.allclose(lookup_table[2, 2], item), "The item (2, 2) is not correct."


@pytest.mark.parametrize(
    "transition_matrix,binary_state,k,observable",
    [
        (
            0.5 * np.array([
                [1, 1j, 0, 0],
                [0, 0, 1, 1j]
            ]),
            "00", 0,  # binary_state, k
            np.array([
                [0, 0],
                [0, 0],
            ])
        )
    ]
)
def test_lookup_table_get_observable(transition_matrix, binary_state, k, observable):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    obs = lookup_table.get_observable(k, utils.binary_state_to_state(binary_state))
    assert np.allclose(obs, observable), "The observable is not correct."
