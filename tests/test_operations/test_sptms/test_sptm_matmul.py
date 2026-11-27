from typing import Any, List, Optional, Sequence

import numpy as np
import pytest
import torch
from pennylane import numpy as pnp
from pennylane.wires import Wires

from matchcake.operations import SptmCompRyRy, SptmCompZX
from matchcake.operations.single_particle_transition_matrices import (
    DisjointSptmOperation,
    SingleParticleTransitionMatrixOperation,
)

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

BATCH_SIZE = 3
# (wires of the left operand, wires of the right operand)
WIRES_CASES = {
    "same_wires": ([0, 1], [0, 1]),
    "same_three_wires": ([1, 2, 3], [1, 2, 3]),
    "right_within_left": ([0, 1, 2, 3], [1, 2]),
    "right_at_the_end_of_left": ([0, 1, 2, 3], [2, 3]),
    "left_within_right": ([1, 2], [0, 1, 2, 3]),
    "disjoint": ([0, 1], [3, 4]),
    "disjoint_reversed": ([3, 4], [0, 1]),
    "partial_overlap": ([0, 1, 2], [2, 3]),
    "partial_overlap_reversed": ([2, 3], [0, 1, 2]),
    "not_starting_at_zero": ([2, 3], [3, 4, 5]),
}
CONTAINED_CASES = ("same_wires", "same_three_wires", "right_within_left", "right_at_the_end_of_left")
NOT_CONTAINED_CASES = tuple(name for name in WIRES_CASES if name not in CONTAINED_CASES)
BATCH_CASES = [
    pytest.param(None, None, id="unbatched"),
    pytest.param(BATCH_SIZE, None, id="left_batched"),
    pytest.param(None, BATCH_SIZE, id="right_batched"),
    pytest.param(BATCH_SIZE, BATCH_SIZE, id="both_batched"),
]
# (wires of the blocks of the left operand, wires of the blocks of the right operand), one block is a plain operation
DISJOINT_CASES = {
    "disjoint_left": ([[0, 1], [3, 4]], [[1, 2, 3]]),
    "disjoint_right": ([[1, 2, 3]], [[0, 1], [3, 4]]),
    "disjoint_right_within_left": ([[0, 1, 2, 3, 4, 5]], [[0, 1], [4, 5]]),
    "disjoint_both": ([[0, 1], [4, 5]], [[1, 2], [3, 4], [5, 6]]),
    "disjoint_right_of_different_sizes": ([[0, 1, 2, 3, 4]], [[0, 1, 2], [3, 4]]),
}


class TestSptmMatmul:
    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _to_backend(array: np.ndarray, backend: str) -> Any:
        if backend == "torch":
            return torch.from_numpy(np.array(array))
        if backend == "autograd":
            return pnp.array(array, requires_grad=True)
        return np.array(array)

    @staticmethod
    def _random_matrix(rng: np.random.Generator, n_wires: int, batch_size: Optional[int]) -> np.ndarray:
        shape = (2 * n_wires, 2 * n_wires) if batch_size is None else (batch_size, 2 * n_wires, 2 * n_wires)
        return rng.normal(size=shape)

    @classmethod
    def _embed(cls, matrix: Any, wires: Sequence[int], first_wire: int, n_wires: int) -> np.ndarray:
        array = cls._to_numpy(matrix)
        start = 2 * (min(wires) - first_wire)
        padded = np.zeros((*array.shape[:-2], 2 * n_wires, 2 * n_wires), dtype=array.dtype)
        padded[...] = np.eye(2 * n_wires)
        padded[..., start : start + array.shape[-2], start : start + array.shape[-1]] = array
        return padded

    @classmethod
    def _dense_product(
        cls,
        left_factors: List[Any],
        left_wires: List[List[int]],
        right_factors: List[Any],
        right_wires: List[List[int]],
    ) -> np.ndarray:
        all_wires = [wire for wires in left_wires + right_wires for wire in wires]
        first_wire, n_wires = min(all_wires), max(all_wires) - min(all_wires) + 1
        result = np.eye(2 * n_wires)
        for factor, wires in zip(left_factors + right_factors, left_wires + right_wires):
            result = result @ cls._embed(factor, wires, first_wire, n_wires)
        return result

    @classmethod
    def _make_operation(
        cls,
        rng: np.random.Generator,
        blocks_wires: List[List[int]],
        backend: str,
        batch_size: Optional[int] = None,
    ) -> SingleParticleTransitionMatrixOperation:
        operations = [
            SingleParticleTransitionMatrixOperation(
                cls._to_backend(cls._random_matrix(rng, len(wires), batch_size), backend), wires=wires
            )
            for wires in blocks_wires
        ]
        operation = SingleParticleTransitionMatrixOperation.from_operations(operations)
        assert operation is not None
        return operation

    @staticmethod
    def _factors(operation: SingleParticleTransitionMatrixOperation) -> List[Any]:
        blocks = operation.blocks
        if blocks is None:
            return [operation.sptm_matrix()]
        return [block.sptm_matrix() for block in blocks]

    @staticmethod
    def _span(*wires_groups: List[int]) -> Wires:
        all_wires = [wire for wires in wires_groups for wire in wires]
        return Wires(range(min(all_wires), max(all_wires) + 1))

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("case_name", list(WIRES_CASES))
    @pytest.mark.parametrize("left_batch_size, right_batch_size", BATCH_CASES)
    def test_matmul_equals_padded_dense_product(self, backend, case_name, left_batch_size, right_batch_size, rng):
        left_wires, right_wires = WIRES_CASES[case_name]
        left = self._make_operation(rng, [left_wires], backend, left_batch_size)
        right = self._make_operation(rng, [right_wires], backend, right_batch_size)
        left_matrix = self._to_numpy(left.sptm_matrix()).copy()
        right_matrix = self._to_numpy(right.sptm_matrix()).copy()
        expected = self._dense_product([left_matrix], [left_wires], [right_matrix], [right_wires])

        product = left @ right

        assert type(product) is SingleParticleTransitionMatrixOperation
        assert product.wires == self._span(left_wires, right_wires)
        assert product.blocks is None
        assert isinstance(product.sptm_matrix(), torch.Tensor if backend == "torch" else np.ndarray)
        assert product.batch_size == (left_batch_size or right_batch_size)
        np.testing.assert_allclose(
            self._to_numpy(product.sptm_matrix()), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_array_equal(self._to_numpy(left.sptm_matrix()), left_matrix)
        np.testing.assert_array_equal(self._to_numpy(right.sptm_matrix()), right_matrix)
        assert left.wires == Wires(left_wires)
        assert right.wires == Wires(right_wires)

    @pytest.mark.parametrize(
        "left_backend, right_backend",
        [
            pytest.param("numpy", "torch", id="numpy_at_torch"),
            pytest.param("torch", "numpy", id="torch_at_numpy"),
        ],
    )
    @pytest.mark.parametrize("case_name", list(WIRES_CASES))
    def test_matmul_of_numpy_and_torch_is_torch(self, left_backend, right_backend, case_name, rng):
        left_wires, right_wires = WIRES_CASES[case_name]
        left = self._make_operation(rng, [left_wires], left_backend)
        right = self._make_operation(rng, [right_wires], right_backend)
        left_matrix = self._to_numpy(left.sptm_matrix()).copy()
        right_matrix = self._to_numpy(right.sptm_matrix()).copy()

        product = left @ right

        assert isinstance(product.sptm_matrix(), torch.Tensor)
        np.testing.assert_allclose(
            self._to_numpy(product.sptm_matrix()),
            self._dense_product([left_matrix], [left_wires], [right_matrix], [right_wires]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_array_equal(self._to_numpy(left.sptm_matrix()), left_matrix)
        np.testing.assert_array_equal(self._to_numpy(right.sptm_matrix()), right_matrix)

    @pytest.mark.parametrize("backend", ["numpy", "torch", "autograd"])
    @pytest.mark.parametrize("case_name", list(DISJOINT_CASES))
    @pytest.mark.parametrize("left_batch_size, right_batch_size", BATCH_CASES)
    def test_matmul_with_disjoint_operations(self, backend, case_name, left_batch_size, right_batch_size, rng):
        left_wires, right_wires = DISJOINT_CASES[case_name]
        left = self._make_operation(rng, left_wires, backend, left_batch_size)
        right = self._make_operation(rng, right_wires, backend, right_batch_size)
        left_factors = [self._to_numpy(factor).copy() for factor in self._factors(left)]
        right_factors = [self._to_numpy(factor).copy() for factor in self._factors(right)]
        expected = self._dense_product(left_factors, left_wires, right_factors, right_wires)

        product = left @ right

        assert type(product) is SingleParticleTransitionMatrixOperation
        assert product.wires == self._span(*left_wires, *right_wires)
        np.testing.assert_allclose(
            self._to_numpy(product.sptm_matrix()), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        for operation, factors in ((left, left_factors), (right, right_factors)):
            for factor_after, factor_before in zip(self._factors(operation), factors):
                np.testing.assert_array_equal(self._to_numpy(factor_after), factor_before)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_matmul_does_not_build_the_dense_matrix_of_a_disjoint_right_operand(self, backend, rng):
        left = self._make_operation(rng, [[0, 1, 2, 3, 4, 5]], backend)
        right = self._make_operation(rng, [[0, 1], [4, 5]], backend)
        assert isinstance(right, DisjointSptmOperation)

        left @ right

        assert right._dense_matrix is None

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_matmul_does_not_modify_the_dense_matrix_of_a_disjoint_left_operand(self, backend, rng):
        left = self._make_operation(rng, [[0, 1], [2, 3]], backend)
        right = self._make_operation(rng, [[1, 2]], backend)
        dense_before = self._to_numpy(left.sptm_matrix()).copy()

        product = left @ right

        assert product.sptm_matrix() is not left.sptm_matrix()
        np.testing.assert_array_equal(self._to_numpy(left.sptm_matrix()), dense_before)

    @pytest.mark.parametrize(
        "left_backend, right_backend",
        [
            pytest.param("autograd", "autograd", id="autograd_at_autograd"),
            pytest.param("numpy", "autograd", id="numpy_at_autograd"),
            pytest.param("autograd", "numpy", id="autograd_at_numpy"),
        ],
    )
    @pytest.mark.parametrize("case_name", list(WIRES_CASES))
    @pytest.mark.parametrize("left_batch_size, right_batch_size", BATCH_CASES)
    def test_matmul_falls_back_to_the_dense_product_for_autograd(
        self, left_backend, right_backend, case_name, left_batch_size, right_batch_size, rng
    ):
        left_wires, right_wires = WIRES_CASES[case_name]
        left = self._make_operation(rng, [left_wires], left_backend, left_batch_size)
        right = self._make_operation(rng, [right_wires], right_backend, right_batch_size)
        left_matrix = self._to_numpy(left.sptm_matrix()).copy()
        right_matrix = self._to_numpy(right.sptm_matrix()).copy()

        product = left @ right

        assert product.wires == self._span(left_wires, right_wires)
        np.testing.assert_allclose(
            self._to_numpy(product.sptm_matrix()),
            self._dense_product([left_matrix], [left_wires], [right_matrix], [right_wires]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_array_equal(self._to_numpy(left.sptm_matrix()), left_matrix)
        np.testing.assert_array_equal(self._to_numpy(right.sptm_matrix()), right_matrix)

    def test_matmul_of_integer_and_float_operations_is_float(self):
        swap = SptmCompZX(wires=[0, 1])
        rotation = SptmCompRyRy(np.array([0.3, 0.4]), wires=[1, 2])
        swap_matrix = swap.sptm_matrix().copy()

        product = swap @ rotation

        assert np.issubdtype(swap_matrix.dtype, np.integer)
        assert np.issubdtype(product.sptm_matrix().dtype, np.floating)
        np.testing.assert_allclose(
            product.sptm_matrix(),
            self._dense_product([swap_matrix], [[0, 1]], [rotation.sptm_matrix()], [[1, 2]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_array_equal(swap.sptm_matrix(), swap_matrix)
        assert np.issubdtype(swap.sptm_matrix().dtype, np.integer)

    @pytest.mark.parametrize("case_name", ["same_wires", "right_within_left", "left_within_right", "partial_overlap"])
    @pytest.mark.parametrize("left_batch_size, right_batch_size", BATCH_CASES)
    def test_matmul_gradcheck(self, case_name, left_batch_size, right_batch_size, rng):
        left_wires, right_wires = WIRES_CASES[case_name]
        left_matrix = torch.from_numpy(self._random_matrix(rng, len(left_wires), left_batch_size)).requires_grad_()
        right_matrix = torch.from_numpy(self._random_matrix(rng, len(right_wires), right_batch_size)).requires_grad_()

        def product(left_input: torch.Tensor, right_input: torch.Tensor) -> torch.Tensor:
            left = SingleParticleTransitionMatrixOperation(left_input, wires=left_wires)
            right = SingleParticleTransitionMatrixOperation(right_input, wires=right_wires)
            return (left @ right).sptm_matrix()

        assert torch.autograd.gradcheck(
            product,
            (left_matrix, right_matrix),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_in_place_matmul_returns_a_new_operation_and_keeps_the_operand(self, backend):
        """``@=`` is ``@``: writing into the stored matrix would change tensors read from it earlier."""
        matrix = SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]).sptm_matrix()
        matrix = torch.as_tensor(matrix) if backend == "torch" else matrix
        left = SingleParticleTransitionMatrixOperation(matrix, wires=[0, 1])
        before = np.array(SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]).sptm_matrix())
        right = SptmCompRyRy(np.array([0.5, 0.1]), wires=[0, 1])
        result = left

        result @= right

        assert result is not left
        np.testing.assert_array_equal(np.asarray(left.sptm_matrix()), before)
        np.testing.assert_allclose(
            np.asarray(result.sptm_matrix()),
            before @ right.sptm_matrix(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [2, 3, 4]])
    def test_blocks_of_a_plain_operation_is_none(self, wires, rng):
        assert self._make_operation(rng, [wires], "numpy").blocks is None

    @pytest.mark.parametrize("backend", ["numpy", "torch", "autograd"])
    @pytest.mark.parametrize(
        "operation_wires, matrix_wires, expected_start",
        [
            pytest.param([0, 1], [0, 1], 0, id="same_wires"),
            pytest.param([2, 3], [0, 1, 2, 3, 4], 4, id="in_the_middle"),
            pytest.param([3, 4, 5], [2, 3, 4, 5], 2, id="matrix_not_starting_at_zero"),
        ],
    )
    def test_column_blocks_gives_the_stored_matrix_and_its_first_column(
        self, backend, operation_wires, matrix_wires, expected_start, rng
    ):
        operation = self._make_operation(rng, [operation_wires], backend)

        column_blocks = operation.column_blocks(Wires(matrix_wires))

        assert len(column_blocks) == 1
        start, block = column_blocks[0]
        assert start == expected_start
        assert block is operation.sptm_matrix()

    def test_continuous_wires_follow_a_wire_map(self):
        """``map_wires`` copies the operation with its attributes, the cached wires among them."""
        operation = SptmCompRyRy(np.array([0.3, 0.5]), wires=[0, 1])
        operation.sptm_matrix()
        mapped = operation.map_wires({0: 2, 1: 3})
        assert mapped.cs_wires.tolist() == [2, 3]
        assert operation.cs_wires.tolist() == [0, 1]
