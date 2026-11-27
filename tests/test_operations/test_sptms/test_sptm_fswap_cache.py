from typing import List

import numpy as np
import pytest

from matchcake.operations.single_particle_transition_matrices import SptmCompZX

WIRES_CASES = [
    [0, 1],
    [1, 0],
    [0, 2],
    [2, 0],
    [3, 4],
    [4, 3],
    [1, 4],
    [4, 1],
    [0, 5],
    [5, 0],
    [2, 7],
]


class TestSptmFSwapCache:
    @staticmethod
    def _reference_matrix(wires: List[int]) -> np.ndarray:
        # the matrix as it was built for every gate before the cache
        wire0, wire1 = sorted(wires)
        size = 2 * (wire1 - wire0 + 1)
        matrix = np.zeros((size, size), dtype=int)
        first_index = 2 * (wire1 - wire0)
        matrix[:2, first_index : first_index + 2] = np.eye(2)
        rows, columns = np.where(np.eye(size, k=-2))
        matrix[rows, columns] = 1
        if wire0 != wires[0]:
            matrix = matrix.T
        return matrix

    @pytest.mark.parametrize("wires", WIRES_CASES)
    def test_matrix_is_the_one_built_without_cache(self, wires: List[int]):
        expected = self._reference_matrix(wires)

        first = np.asarray(SptmCompZX(wires=wires).sptm_matrix())
        second = np.asarray(SptmCompZX(wires=wires).sptm_matrix())

        assert first.shape == expected.shape
        assert first.dtype == expected.dtype
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)

    @pytest.mark.parametrize("wires", WIRES_CASES)
    def test_matrix_is_built_when_the_cache_is_empty(self, wires: List[int], monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(SptmCompZX, "_MATRIX_CACHE", {})

        matrix = np.asarray(SptmCompZX(wires=wires).sptm_matrix())

        assert list(SptmCompZX._MATRIX_CACHE) == [(abs(wires[0] - wires[1]), wires[0] > wires[1])]
        np.testing.assert_array_equal(matrix, self._reference_matrix(wires))
        np.testing.assert_array_equal(np.asarray(SptmCompZX(wires=wires).sptm_matrix()), matrix)
        assert len(SptmCompZX._MATRIX_CACHE) == 1

    @pytest.mark.parametrize("wires, other_wires", [([0, 3], [2, 5]), ([3, 0], [5, 2]), ([0, 1], [6, 7])])
    def test_gates_with_the_same_distance_and_order_share_one_cache_entry(
        self, wires: List[int], other_wires: List[int], monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(SptmCompZX, "_MATRIX_CACHE", {})

        first = SptmCompZX(wires=wires)
        second = SptmCompZX(wires=other_wires)

        assert len(SptmCompZX._MATRIX_CACHE) == 1
        np.testing.assert_array_equal(np.asarray(first.sptm_matrix()), np.asarray(second.sptm_matrix()))

    def test_both_orders_of_distant_wires_have_their_own_cache_entry(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(SptmCompZX, "_MATRIX_CACHE", {})

        forward = np.asarray(SptmCompZX(wires=[0, 2]).sptm_matrix())
        backward = np.asarray(SptmCompZX(wires=[2, 0]).sptm_matrix())

        assert set(SptmCompZX._MATRIX_CACHE) == {(2, False), (2, True)}
        np.testing.assert_array_equal(backward, forward.T)
        assert not np.array_equal(backward, forward)

    @pytest.mark.parametrize("wires", [[0, 1], [0, 3], [3, 0]])
    def test_two_gates_do_not_share_memory(self, wires: List[int]):
        first = SptmCompZX(wires=wires)
        second = SptmCompZX(wires=wires)
        expected = self._reference_matrix(wires)

        assert not np.shares_memory(first.sptm_matrix(), second.sptm_matrix())
        first.sptm_matrix()[...] = 7

        np.testing.assert_array_equal(np.asarray(second.sptm_matrix()), expected)
        np.testing.assert_array_equal(np.asarray(SptmCompZX(wires=wires).sptm_matrix()), expected)

    @pytest.mark.parametrize("wires", [[0, 1], [1, 0], [5, 6]])
    def test_a_gate_on_adjacent_wires_has_empty_parameters(self, wires: List[int]):
        gate = SptmCompZX(wires=wires)

        assert isinstance(gate._params, np.ndarray)
        assert gate._params.shape == (0,)
        assert gate._stored_matrix is not None
        assert gate.batch_size is None
        np.testing.assert_array_equal(gate.data[0], self._reference_matrix(wires))

    @pytest.mark.parametrize("wires", [[0, 2], [2, 0], [1, 5]])
    def test_a_gate_on_distant_wires_has_no_parameters(self, wires: List[int]):
        gate = SptmCompZX(wires=wires)

        assert gate._params is None
        assert gate.batch_size is None
