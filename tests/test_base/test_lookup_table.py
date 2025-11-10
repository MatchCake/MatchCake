import numpy as np
import pytest

from matchcake import utils
from matchcake.base.lookup_table import NonInteractingFermionicLookupTable

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestNonInteractingFermionicLookupTable:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

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
    def test_lookup_table_call(self, transition_matrix, binary_state, observable):
        lookup_table = NonInteractingFermionicLookupTable(transition_matrix, show_progress=True)
        obs = lookup_table.compute_observables_of_target_states(utils.binary_string_to_vector(binary_state))
        np.testing.assert_allclose(
            obs.squeeze(),
            observable,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item00(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[0],
            TBT,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 0 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item01(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[1],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 1 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item02(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[2],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 2 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item10(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[3],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 3 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item11(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[4],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 4 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item12(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[5],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 5 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item20(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[6],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 6 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item21(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
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
        np.testing.assert_allclose(
            lookup_table[7],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 7 is not correct.",
        )

    @pytest.mark.parametrize("n_particles", list(range(2, 5)))
    def test_lookup_table_item22(self, n_particles):
        transition_matrix = np.random.rand(n_particles, 2 * n_particles)
        lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
        item = np.eye(transition_matrix.shape[-1])
        np.testing.assert_allclose(
            lookup_table[2, 2],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item (2, 2) is not correct.",
        )
        np.testing.assert_allclose(
            lookup_table[8],
            item,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
            err_msg="The item 8 is not correct.",
        )

    @pytest.mark.parametrize("binary_state", ["00", "01", "10", "11"])
    def test_lookup_table_observable_form(self, binary_state):
        transition_matrix = np.random.rand(len(binary_state), 2 * len(binary_state))
        lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
        binary_state = utils.binary_string_to_vector(binary_state)
        hamming_weight = np.sum(binary_state, dtype=int)

        obs = lookup_table.compute_observable_of_target_state(binary_state)
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

    def test_shape(self):
        lt = NonInteractingFermionicLookupTable(np.random.rand(2, 4))
        assert lt.shape == (3, 3)
