from typing import Any, List, Tuple

import numpy as np
import pytest
import torch
from pennylane import numpy as pnp

from matchcake.operations import SptmCompZX
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

MATRIX_SIZE = 14
BATCH_SIZE = 3
# (first column, block size) of every block
BLOCK_LAYOUTS = {
    "one_block": [(4, 4)],
    "two_same_size": [(0, 4), (8, 4)],
    "three_same_size": [(0, 4), (4, 4), (10, 4)],
    "different_sizes": [(0, 2), (2, 4), (8, 6)],
    "same_and_different_sizes": [(0, 4), (4, 2), (6, 4), (12, 2)],
    "unsorted_starts": [(10, 4), (0, 4), (6, 2)],
    "whole_matrix": [(0, MATRIX_SIZE)],
}
BLOCK_BATCHINGS = ("unbatched", "batched", "mixed")
GRADCHECK_MATRIX_SIZE = 6
GRADCHECK_BATCH_SIZE = 2
GRADCHECK_LAYOUTS = {
    "one_block": [(2, 4)],
    "two_same_size": [(0, 2), (4, 2)],
    "different_sizes": [(0, 2), (2, 4)],
}
DTYPE_CASES = [
    pytest.param("int64", "int64", "int64", id="int_matrix_int_block"),
    pytest.param("int64", "float64", "float64", id="int_matrix_float_block"),
    pytest.param("float64", "int64", "float64", id="float_matrix_int_block"),
    pytest.param("float32", "float64", "float64", id="float32_matrix_float64_block"),
    pytest.param("float64", "float32", "float64", id="float64_matrix_float32_block"),
    pytest.param("float32", "float32", "float32", id="float32_matrix_float32_block"),
]


class TestSptmMultiplyColumns:
    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _to_backend(array: np.ndarray, backend: str) -> Any:
        if backend == "torch":
            return torch.from_numpy(np.array(array))
        return np.array(array)

    @staticmethod
    def _make_blocks(
        layout: List[Tuple[int, int]],
        batching: str,
        rng: np.random.Generator,
        batch_size: int = BATCH_SIZE,
    ) -> List[Tuple[int, np.ndarray]]:
        blocks = []
        for index, (start, size) in enumerate(layout):
            is_batched = batching == "batched" or (batching == "mixed" and index % 2 == 0)
            shape = (batch_size, size, size) if is_batched else (size, size)
            blocks.append((start, rng.normal(size=shape)))
        return blocks

    @classmethod
    def _dense_product(cls, matrix: Any, blocks: List[Tuple[int, Any]]) -> np.ndarray:
        result = cls._to_numpy(matrix).copy()
        size = result.shape[-1]
        for start, block in blocks:
            block_array = cls._to_numpy(block)
            block_size = block_array.shape[-1]
            padded = np.zeros((*block_array.shape[:-2], size, size), dtype=block_array.dtype)
            padded[...] = np.eye(size)
            padded[..., start : start + block_size, start : start + block_size] = block_array
            result = result @ padded
        return result

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("matrix_is_batched", [False, True])
    @pytest.mark.parametrize("layout_name", list(BLOCK_LAYOUTS))
    @pytest.mark.parametrize("batching", BLOCK_BATCHINGS)
    def test_equals_dense_product_with_padded_blocks(self, backend, matrix_is_batched, layout_name, batching, rng):
        matrix_shape = (BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE) if matrix_is_batched else (MATRIX_SIZE, MATRIX_SIZE)
        matrix = rng.normal(size=matrix_shape)
        blocks = self._make_blocks(BLOCK_LAYOUTS[layout_name], batching, rng)
        expected = self._dense_product(matrix, blocks)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            self._to_backend(matrix, backend),
            [(start, self._to_backend(block, backend)) for start, block in blocks],
        )

        assert isinstance(result, torch.Tensor if backend == "torch" else np.ndarray)
        assert tuple(result.shape) == expected.shape
        np.testing.assert_allclose(
            self._to_numpy(result), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("layout_name", ["one_block", "three_same_size", "same_and_different_sizes"])
    def test_batched_blocks_make_an_unbatched_matrix_batched(self, backend, layout_name, rng):
        matrix = rng.normal(size=(MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS[layout_name], "batched", rng)
        matrix_input = self._to_backend(matrix, backend)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            matrix_input,
            [(start, self._to_backend(block, backend)) for start, block in blocks],
        )

        assert tuple(result.shape) == (BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE)
        np.testing.assert_array_equal(self._to_numpy(matrix_input), matrix)
        np.testing.assert_allclose(
            self._to_numpy(result),
            self._dense_product(matrix, blocks),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "matrix_backend, block_backends",
        [
            pytest.param("numpy", ("torch", "torch", "torch"), id="numpy_matrix_torch_blocks"),
            pytest.param("torch", ("numpy", "numpy", "numpy"), id="torch_matrix_numpy_blocks"),
            pytest.param("numpy", ("numpy", "torch", "numpy"), id="numpy_matrix_mixed_blocks"),
            pytest.param("torch", ("torch", "numpy", "torch"), id="torch_matrix_mixed_blocks"),
        ],
    )
    @pytest.mark.parametrize("layout_name", ["three_same_size", "different_sizes"])
    @pytest.mark.parametrize("batching", BLOCK_BATCHINGS)
    def test_torch_wins_over_numpy(self, matrix_backend, block_backends, layout_name, batching, rng):
        matrix = rng.normal(size=(MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS[layout_name], batching, rng)
        expected = self._dense_product(matrix, blocks)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            self._to_backend(matrix, matrix_backend),
            [(start, self._to_backend(block, backend)) for (start, block), backend in zip(blocks, block_backends)],
        )

        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(
            self._to_numpy(result), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("matrix_dtype, block_dtype, expected_dtype", DTYPE_CASES)
    @pytest.mark.parametrize("layout_name", ["one_block", "two_same_size", "different_sizes"])
    def test_dtype_is_the_promotion_of_all_dtypes(
        self, backend, matrix_dtype, block_dtype, expected_dtype, layout_name, rng
    ):
        matrix = rng.integers(-3, 4, size=(MATRIX_SIZE, MATRIX_SIZE)).astype(matrix_dtype)
        blocks = [
            (start, rng.integers(-3, 4, size=(size, size)).astype(block_dtype))
            for start, size in BLOCK_LAYOUTS[layout_name]
        ]
        expected = self._dense_product(matrix, blocks)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            self._to_backend(matrix, backend),
            [(start, self._to_backend(block, backend)) for start, block in blocks],
        )

        assert str(result.dtype).replace("torch.", "") == expected_dtype
        np.testing.assert_allclose(
            self._to_numpy(result), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("starts", [(2,), (0, 6), (0, 4, 10)])
    def test_integer_fermionic_swap_blocks_stay_integer(self, backend, starts):
        swap_block = SptmCompZX(wires=[0, 1]).sptm_matrix()
        assert np.issubdtype(swap_block.dtype, np.integer)
        matrix = np.arange(MATRIX_SIZE**2, dtype=np.int64).reshape(MATRIX_SIZE, MATRIX_SIZE)
        blocks = [(start, swap_block) for start in starts]
        expected = self._dense_product(matrix, blocks)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            self._to_backend(matrix, backend),
            [(start, self._to_backend(block, backend)) for start, block in blocks],
        )

        assert str(result.dtype).replace("torch.", "") == "int64"
        np.testing.assert_array_equal(self._to_numpy(result), expected)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("layout_name", ["one_block", "three_same_size", "different_sizes"])
    def test_columns_are_written_in_place_when_no_promotion_is_needed(self, backend, layout_name, rng):
        matrix = rng.normal(size=(MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS[layout_name], "unbatched", rng)
        expected = self._dense_product(matrix, blocks)
        matrix_input = self._to_backend(matrix, backend)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            matrix_input,
            [(start, self._to_backend(block, backend)) for start, block in blocks],
        )

        assert result is matrix_input
        np.testing.assert_allclose(
            self._to_numpy(matrix_input), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("layout_name", ["one_block", "three_same_size", "different_sizes"])
    def test_blocks_and_untouched_columns_are_left_unchanged(self, backend, layout_name, rng):
        matrix = rng.normal(size=(BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS[layout_name], "mixed", rng)
        block_inputs = [(start, self._to_backend(block, backend)) for start, block in blocks]
        touched = np.zeros(MATRIX_SIZE, dtype=bool)
        for start, block in blocks:
            touched[start : start + block.shape[-1]] = True

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            self._to_backend(matrix, backend), block_inputs
        )

        np.testing.assert_array_equal(self._to_numpy(result)[..., ~touched], matrix[..., ~touched])
        for (_, block_input), (_, block) in zip(block_inputs, blocks):
            np.testing.assert_array_equal(self._to_numpy(block_input), block)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_dtype_promotion_leaves_the_input_matrix_unchanged(self, backend, rng):
        matrix = rng.integers(-3, 4, size=(MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS["two_same_size"], "unbatched", rng)
        matrix_input = self._to_backend(matrix, backend)

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            matrix_input,
            [(start, self._to_backend(block, backend)) for start, block in blocks],
        )

        assert result is not matrix_input
        np.testing.assert_array_equal(self._to_numpy(matrix_input), matrix)

    def test_backend_promotion_leaves_the_input_matrix_unchanged(self, rng):
        matrix = rng.normal(size=(MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS["two_same_size"], "unbatched", rng)
        matrix_input = np.array(matrix)

        SingleParticleTransitionMatrixOperation.multiply_columns(
            matrix_input,
            [(start, torch.from_numpy(block)) for start, block in blocks],
        )

        np.testing.assert_array_equal(matrix_input, matrix)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_no_blocks_returns_the_matrix(self, backend, rng):
        matrix = rng.normal(size=(MATRIX_SIZE, MATRIX_SIZE))

        result = SingleParticleTransitionMatrixOperation.multiply_columns(self._to_backend(matrix, backend), [])

        np.testing.assert_array_equal(self._to_numpy(result), matrix)

    @pytest.mark.parametrize(
        "matrix_kind, block_kind",
        [
            pytest.param("autograd", "autograd", id="autograd_matrix_autograd_block"),
            pytest.param("autograd", "numpy", id="autograd_matrix_numpy_block"),
            pytest.param("numpy", "autograd", id="numpy_matrix_autograd_block"),
            pytest.param("torch", "autograd", id="torch_matrix_autograd_block"),
            pytest.param("autograd", "torch", id="autograd_matrix_torch_block"),
        ],
    )
    def test_returns_none_for_pennylane_numpy(self, matrix_kind, block_kind, rng):
        converters = {
            "autograd": lambda array: pnp.array(array, requires_grad=True),
            "numpy": np.array,
            "torch": torch.from_numpy,
        }
        matrix = rng.normal(size=(MATRIX_SIZE, MATRIX_SIZE))
        blocks = self._make_blocks(BLOCK_LAYOUTS["two_same_size"], "unbatched", rng)
        matrix_input = converters[matrix_kind](matrix.copy())

        result = SingleParticleTransitionMatrixOperation.multiply_columns(
            matrix_input,
            [(start, converters[block_kind](block)) for start, block in blocks],
        )

        assert result is None
        np.testing.assert_array_equal(self._to_numpy(matrix_input), matrix)

    @pytest.mark.parametrize("matrix_is_batched", [False, True])
    @pytest.mark.parametrize("layout_name", list(BLOCK_LAYOUTS))
    @pytest.mark.parametrize("batching", BLOCK_BATCHINGS)
    def test_torch_leaf_requiring_grad_is_not_modified(self, matrix_is_batched, layout_name, batching, rng):
        matrix_shape = (BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE) if matrix_is_batched else (MATRIX_SIZE, MATRIX_SIZE)
        matrix = rng.normal(size=matrix_shape)
        blocks = self._make_blocks(BLOCK_LAYOUTS[layout_name], batching, rng)
        weights = rng.normal(size=(BATCH_SIZE, MATRIX_SIZE, MATRIX_SIZE))
        matrix_input = torch.from_numpy(matrix.copy()).requires_grad_()
        block_inputs = [(start, torch.from_numpy(block).requires_grad_()) for start, block in blocks]

        result = SingleParticleTransitionMatrixOperation.multiply_columns(matrix_input, block_inputs)

        assert result is not matrix_input
        assert result.requires_grad
        np.testing.assert_array_equal(self._to_numpy(matrix_input), matrix)
        np.testing.assert_allclose(
            self._to_numpy(result),
            self._dense_product(matrix, blocks),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        # gradient of sum(weights * (matrix @ padded)) with respect to matrix is weights @ padded^T
        (result * torch.from_numpy(weights)).sum().backward()
        expected_grad = self._dense_product(weights, [(start, np.swapaxes(block, -1, -2)) for start, block in blocks])
        if not matrix_is_batched:
            expected_grad = expected_grad.sum(axis=0)
        assert matrix_input.grad is not None
        np.testing.assert_allclose(
            self._to_numpy(matrix_input.grad), expected_grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        for _, block_input in block_inputs:
            assert block_input.grad is not None

    @pytest.mark.parametrize("matrix_kind", ["leaf", "non_leaf", "constant"])
    @pytest.mark.parametrize("matrix_is_batched", [False, True])
    @pytest.mark.parametrize("layout_name", list(GRADCHECK_LAYOUTS))
    @pytest.mark.parametrize("batching", BLOCK_BATCHINGS)
    def test_gradcheck_with_respect_to_matrix_and_blocks(
        self, matrix_kind, matrix_is_batched, layout_name, batching, rng
    ):
        size = GRADCHECK_MATRIX_SIZE
        matrix_shape = (GRADCHECK_BATCH_SIZE, size, size) if matrix_is_batched else (size, size)
        matrix = torch.from_numpy(rng.normal(size=matrix_shape))
        blocks = self._make_blocks(GRADCHECK_LAYOUTS[layout_name], batching, rng, GRADCHECK_BATCH_SIZE)
        starts = [start for start, _ in blocks]
        block_inputs = tuple(torch.from_numpy(block).requires_grad_() for _, block in blocks)

        def product_leaf(matrix_input: torch.Tensor, *block_args: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(
                SingleParticleTransitionMatrixOperation.multiply_columns(matrix_input, list(zip(starts, block_args)))
            )

        def product_non_leaf(matrix_input: torch.Tensor, *block_args: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(
                SingleParticleTransitionMatrixOperation.multiply_columns(
                    2.0 * matrix_input, list(zip(starts, block_args))
                )
            )

        def product_constant(*block_args: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(
                SingleParticleTransitionMatrixOperation.multiply_columns(matrix.clone(), list(zip(starts, block_args)))
            )

        if matrix_kind == "constant":
            function: Any = product_constant
            inputs: Tuple[torch.Tensor, ...] = block_inputs
        else:
            function = product_leaf if matrix_kind == "leaf" else product_non_leaf
            inputs = (matrix.clone().requires_grad_(), *block_inputs)

        assert torch.autograd.gradcheck(
            function,
            inputs,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    @pytest.mark.parametrize("layout_name", list(GRADCHECK_LAYOUTS))
    def test_gradcheck_of_chained_products(self, layout_name, rng):
        size = GRADCHECK_MATRIX_SIZE
        matrix = torch.from_numpy(rng.normal(size=(size, size))).requires_grad_()
        first_blocks = self._make_blocks(GRADCHECK_LAYOUTS[layout_name], "unbatched", rng)
        second_blocks = self._make_blocks([(0, 4), (4, 2)], "mixed", rng, GRADCHECK_BATCH_SIZE)
        starts = [start for start, _ in first_blocks]
        second_starts = [start for start, _ in second_blocks]
        n_first = len(first_blocks)
        block_inputs = tuple(torch.from_numpy(block).requires_grad_() for _, block in first_blocks + second_blocks)

        def chained_product(matrix_input: torch.Tensor, *block_args: torch.Tensor) -> torch.Tensor:
            result = SingleParticleTransitionMatrixOperation.multiply_columns(
                matrix_input, list(zip(starts, block_args[:n_first]))
            )
            return torch.as_tensor(
                SingleParticleTransitionMatrixOperation.multiply_columns(
                    result, list(zip(second_starts, block_args[n_first:]))
                )
            )

        assert torch.autograd.gradcheck(
            chained_product,
            (matrix, *block_inputs),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("n_blocks", [1, 2])
    def test_a_batch_of_one_is_broadcast_to_the_batch_of_the_blocks(self, backend, n_blocks):
        generator = np.random.default_rng(TEST_SEED)
        matrix = generator.normal(size=(1, 8, 8))
        blocks = [(4 * index, generator.normal(size=(3, 4, 4))) for index in range(n_blocks)]
        expected = np.broadcast_to(matrix, (3, 8, 8)).copy()
        for start, block in blocks:
            expected[..., :, start : start + 4] = expected[..., :, start : start + 4] @ block
        if backend == "torch":
            matrix, blocks = torch.from_numpy(matrix), [(start, torch.from_numpy(block)) for start, block in blocks]
        result = SingleParticleTransitionMatrixOperation.multiply_columns(matrix, blocks)
        np.testing.assert_allclose(
            np.asarray(result), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert tuple(matrix.shape) == (1, 8, 8)
