import itertools

import numpy as np
import pennylane as qml
import pytest

from matchcake import utils
from matchcake.base.lookup_table import NonInteractingFermionicLookupTable
from matchcake.circuits import random_sptm_operations_generator

from .configs import (
    ATOL_MATRIX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .test_nif_device import devices_init

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "transition_matrix, binary_state",
    [(np.random.rand(len(s), 2 * len(s)), s) for s in ["00", "01", "10", "11"] for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_observable_form(transition_matrix, binary_state):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    binary_state = utils.binary_string_to_vector(binary_state)
    hamming_weight = np.sum(binary_state, dtype=int)

    obs = lookup_table.get_observable(0, binary_state)
    np.testing.assert_allclose(
        obs.shape,
        (2 * hamming_weight + 2, 2 * hamming_weight + 2),
        err_msg="The observable has the wrong shape.",
    )
    np.testing.assert_allclose(
        obs + obs.T,
        np.zeros_like(obs),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The observable is not symmetric.",
    )
    np.testing.assert_allclose(
        np.diagonal(obs),
        np.zeros(obs.shape[0]),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The diagonal of the observable is not zero.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item00(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    TBT = transition_matrix @ B @ transition_matrix.T
    np.testing.assert_allclose(
        lookup_table[0, 0],
        TBT,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (0, 0) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item01(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = transition_matrix @ B @ np.conjugate(transition_matrix.T)
    np.testing.assert_allclose(
        lookup_table[0, 1],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (0, 1) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item02(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = transition_matrix @ B
    np.testing.assert_allclose(
        lookup_table[0, 2],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (0, 2) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item10(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B @ transition_matrix.T
    np.testing.assert_allclose(
        lookup_table[1, 0],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (1, 0) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item11(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B @ np.conjugate(transition_matrix.T)
    np.testing.assert_allclose(
        lookup_table[1, 1],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (1, 1) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item12(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = np.conjugate(transition_matrix) @ B
    np.testing.assert_allclose(
        lookup_table[1, 2],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (1, 2) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item20(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = B @ transition_matrix.T
    np.testing.assert_allclose(
        lookup_table[2, 0],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (2, 0) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item21(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    B = utils.get_block_diagonal_matrix(transition_matrix.shape[0])
    item = B @ np.conjugate(transition_matrix.T)
    np.testing.assert_allclose(
        lookup_table[2, 1],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (2, 1) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [np.random.rand(n, 2 * n) for n in range(2, 5) for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_lookup_table_item22(transition_matrix):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    item = np.eye(transition_matrix.shape[-1])
    np.testing.assert_allclose(
        lookup_table[2, 2],
        item,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
        err_msg="The item (2, 2) is not correct.",
    )


@pytest.mark.parametrize(
    "transition_matrix,binary_state,k,observable",
    [
        #
        (
            0.5 * np.array([[1, 1j, 0, 0], [0, 0, 1, 1j]]),
            "00",
            0,  # binary_state, k
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        ),
        #
        (
            0.5 * np.array([[0, 0, 1, 1j], [1, 1j, 0, 0]]),
            "01",
            0,  # binary_state, k
            np.array(
                [
                    [0, 1, 0, 1],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [-1, 0, -1, 0],
                ]
            ),
        ),
        #
    ],
)
def test_lookup_table_get_observable(transition_matrix, binary_state, k, observable):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    obs = lookup_table.get_observable(k, utils.binary_string_to_vector(binary_state))
    np.testing.assert_allclose(
        obs,
        observable,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "transition_matrix,binary_state,observable",
    [
        #
        (
            0.5 * np.array([[1, 1j, 0, 0], [0, 0, 1, 1j]]),
            "00",
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        ),
        #
        (
            0.5 * np.array([[0, 0, 1, 1j], [1, 1j, 0, 0]]),
            "01",
            np.array(
                [
                    [0, 1, 0, 1],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [-1, 0, -1, 0],
                ]
            ),
        ),
        #
    ],
)
def test_lookup_table_compute_observable_of_target_state(transition_matrix, binary_state, observable):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    obs = lookup_table.compute_observable_of_target_state(utils.binary_string_to_vector(binary_state))
    np.testing.assert_allclose(
        obs,
        observable,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "transition_matrix,binary_state,observable",
    [
        #
        (
            0.5 * np.array([[1, 1j, 0, 0], [0, 0, 1, 1j]]),
            "00",
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        ),
        #
        (
            0.5 * np.array([[0, 0, 1, 1j], [1, 1j, 0, 0]]),
            "01",
            np.array(
                [
                    [0, 1, 0, 1],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [-1, 0, -1, 0],
                ]
            ),
        ),
        #
    ],
)
def test_lookup_table_compute_observables_of_target_states(transition_matrix, binary_state, observable):
    lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
    obs = lookup_table.compute_observables_of_target_states(utils.binary_string_to_vector(binary_state))
    np.testing.assert_allclose(
        obs.squeeze(),
        observable,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "operations_generator, num_wires",
    [
        (
            random_sptm_operations_generator(num_gates, np.arange(num_wires), batch_size=batch_size),
            num_wires,
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
        for batch_size in [None, 16]
    ],
)
def test_lookup_table_compute_observable_of_target_states_rn_circuits(operations_generator, num_wires):
    nif_device, _ = devices_init(wires=num_wires)
    nif_device.execute_generator(operations_generator)

    lookup_table = nif_device.lookup_table
    target_states = np.array(list(itertools.product([0, 1], repeat=num_wires)))

    obs_list = [
        lookup_table.compute_observable_of_target_state(nif_device.get_sparse_or_dense_state(), target_state)
        for target_state in target_states.copy()
    ]
    loop_obs = qml.math.stack(obs_list, axis=0)
    vec_obs = lookup_table.compute_observables_of_target_states(
        nif_device.get_sparse_or_dense_state(), target_states.copy()
    )
    np.testing.assert_allclose(
        vec_obs,
        loop_obs,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "operations_generator, num_wires",
    [
        (
            random_sptm_operations_generator(num_gates, np.arange(num_wires), batch_size=batch_size),
            num_wires,
        )
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 10 * num_wires]
        for batch_size in [None, 16]
    ],
)
def test_lookup_table_compute_observable_of_target_states_rn_circuits_one_tstate(operations_generator, num_wires):
    nif_device, _ = devices_init(wires=num_wires)
    nif_device.execute_generator(operations_generator)

    lookup_table = nif_device.lookup_table
    target_state = np.random.randint(0, 2, size=num_wires)

    obs_single = lookup_table.compute_observable_of_target_state(
        nif_device.get_sparse_or_dense_state(), target_state.copy()
    )
    vec_obs = lookup_table.compute_observables_of_target_states(
        nif_device.get_sparse_or_dense_state(), target_state.copy()
    )
    np.testing.assert_allclose(
        vec_obs,
        obs_single,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
