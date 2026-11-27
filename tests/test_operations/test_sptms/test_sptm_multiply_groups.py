from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRyRy,
    SptmCompRzRz,
)

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

N_COLUMNS = 12
BATCH_SIZE = 3
# for each step, the (first column, block size) of every block, the blocks of a step being disjoint
STEPS_CASES = {
    "one_block": [[(2, 4)]],
    "one_step": [[(0, 4), (4, 4), (8, 4)]],
    "overlapping_steps": [[(0, 4), (4, 4)], [(2, 4), (6, 4)], [(0, 4), (8, 4)]],
    "two_sizes": [[(0, 4)], [(4, 6)], [(2, 4), (6, 6)], [(0, 6), (6, 6)]],
    "whole_matrix": [[(0, 12)], [(4, 4)]],
    "unsorted_blocks": [[(8, 4), (0, 4)], [(6, 4), (2, 4)]],
}
BACKEND_PAIRS = [
    pytest.param("numpy", "numpy", id="numpy_matrix_numpy_blocks"),
    pytest.param("torch", "torch", id="torch_matrix_torch_blocks"),
    pytest.param("torch", "numpy", id="torch_matrix_numpy_blocks"),
    pytest.param("numpy", "torch", id="numpy_matrix_torch_blocks"),
]


class TestSptmMultiplyGroups:
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
    def _random_orthogonal(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
        orthogonal, _ = np.linalg.qr(rng.normal(size=shape))
        return orthogonal

    @classmethod
    def _random_groups(
        cls,
        rng: np.random.Generator,
        steps: Sequence[Sequence[Tuple[int, int]]],
        batch_size: Optional[int] = None,
        orthogonal: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        groups = []
        for step in steps:
            for size in sorted({size for _, size in step}):
                starts = [start for start, block_size in step if block_size == size]
                shape = (len(starts), size, size) if batch_size is None else (batch_size, len(starts), size, size)
                blocks = cls._random_orthogonal(rng, shape) if orthogonal else rng.normal(size=shape)
                indices = np.array(starts)[:, None] + np.arange(size)[None, :]
                groups.append((indices, blocks))
        return groups

    @classmethod
    def _dense_product(cls, matrix: np.ndarray, groups: Sequence[Tuple[Any, Any]]) -> np.ndarray:
        result = np.array(matrix, dtype=np.float64)
        size = result.shape[-1]
        for indices, stacked in groups:
            indices, stacked = cls._to_numpy(indices), cls._to_numpy(stacked)
            for position, columns in enumerate(indices):
                block = stacked[..., position, :, :]
                padded = np.zeros((*block.shape[:-2], size, size))
                padded[...] = np.eye(size)
                padded[..., columns[0] : columns[-1] + 1, columns[0] : columns[-1] + 1] = block
                result = result @ padded
        return result

    @staticmethod
    def _torch_dense_product(matrix: torch.Tensor, groups: Sequence[Tuple[Any, torch.Tensor]]) -> torch.Tensor:
        result = matrix
        size = matrix.shape[-1]
        for indices, stacked in groups:
            for position in range(len(indices)):
                start, stop = int(indices[position][0]), int(indices[position][-1]) + 1
                block = stacked[..., position, :, :]
                eye = torch.eye(size, dtype=block.dtype).expand(*block.shape[:-2], size, size)
                padded = torch.cat(
                    [
                        eye[..., :, :start],
                        torch.cat([eye[..., :start, start:stop], block, eye[..., stop:, start:stop]], dim=-2),
                        eye[..., :, stop:],
                    ],
                    dim=-1,
                )
                result = result @ padded
        return result

    @classmethod
    def _groups_to_backend(cls, groups: Sequence[Tuple[np.ndarray, np.ndarray]], backend: str) -> List[Tuple[Any, Any]]:
        return [(cls._to_backend(indices, backend), cls._to_backend(stacked, backend)) for indices, stacked in groups]

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("matrix_backend, blocks_backend", BACKEND_PAIRS)
    @pytest.mark.parametrize("steps_name", list(STEPS_CASES))
    @pytest.mark.parametrize("orthogonal", [True, False])
    def test_product_is_the_ordered_product_of_the_padded_blocks(
        self, matrix_backend: str, blocks_backend: str, steps_name: str, orthogonal: bool, rng: np.random.Generator
    ):
        matrix_values = rng.normal(size=(N_COLUMNS, N_COLUMNS))
        groups = self._random_groups(rng, STEPS_CASES[steps_name], orthogonal=orthogonal)
        matrix = self._to_backend(matrix_values, matrix_backend)

        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            matrix, self._groups_to_backend(groups, blocks_backend)
        )

        is_torch = "torch" in (matrix_backend, blocks_backend)
        assert isinstance(product, torch.Tensor if is_torch else np.ndarray)
        assert tuple(product.shape) == (N_COLUMNS, N_COLUMNS)
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(matrix_values, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("matrix_backend, blocks_backend", BACKEND_PAIRS)
    def test_the_matrix_is_not_written_into(self, matrix_backend: str, blocks_backend: str, rng: np.random.Generator):
        matrix_values = rng.normal(size=(N_COLUMNS, N_COLUMNS))
        groups = self._random_groups(rng, STEPS_CASES["overlapping_steps"])
        blocks_values = [stacked.copy() for _, stacked in groups]
        matrix = self._to_backend(matrix_values, matrix_backend)
        backend_groups = self._groups_to_backend(groups, blocks_backend)

        product = SingleParticleTransitionMatrixOperation.multiply_groups(matrix, backend_groups)
        product[...] = 0.0

        np.testing.assert_array_equal(self._to_numpy(matrix), matrix_values)
        for (_, stacked), values in zip(backend_groups, blocks_values):
            np.testing.assert_array_equal(self._to_numpy(stacked), values)

    @pytest.mark.parametrize("matrix_backend, blocks_backend", BACKEND_PAIRS)
    @pytest.mark.parametrize(
        "matrix_shape",
        [
            pytest.param((N_COLUMNS, N_COLUMNS), id="unbatched_matrix"),
            pytest.param((1, N_COLUMNS, N_COLUMNS), id="matrix_batch_of_one"),
            pytest.param((BATCH_SIZE, N_COLUMNS, N_COLUMNS), id="batched_matrix"),
        ],
    )
    @pytest.mark.parametrize("steps_name", ["one_block", "overlapping_steps", "two_sizes"])
    def test_batched_blocks_are_broadcast_with_the_matrix(
        self,
        matrix_backend: str,
        blocks_backend: str,
        matrix_shape: Tuple[int, ...],
        steps_name: str,
        rng: np.random.Generator,
    ):
        matrix_values = rng.normal(size=matrix_shape)
        groups = self._random_groups(rng, STEPS_CASES[steps_name], batch_size=BATCH_SIZE)
        matrix = self._to_backend(matrix_values, matrix_backend)

        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            matrix, self._groups_to_backend(groups, blocks_backend)
        )

        assert tuple(product.shape) == (BATCH_SIZE, N_COLUMNS, N_COLUMNS)
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(matrix_values, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_array_equal(self._to_numpy(matrix), matrix_values)

    @pytest.mark.parametrize("matrix_backend, blocks_backend", BACKEND_PAIRS)
    def test_a_batched_matrix_is_multiplied_by_unbatched_blocks(
        self, matrix_backend: str, blocks_backend: str, rng: np.random.Generator
    ):
        matrix_values = rng.normal(size=(BATCH_SIZE, N_COLUMNS, N_COLUMNS))
        groups = self._random_groups(rng, STEPS_CASES["two_sizes"])

        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            self._to_backend(matrix_values, matrix_backend), self._groups_to_backend(groups, blocks_backend)
        )

        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(matrix_values, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_without_groups_the_matrix_is_returned(self, backend: str, rng: np.random.Generator):
        matrix = self._to_backend(rng.normal(size=(N_COLUMNS, N_COLUMNS)), backend)

        assert SingleParticleTransitionMatrixOperation.multiply_groups(matrix, []) is matrix

    @pytest.mark.parametrize("matrix_backend, blocks_backend", BACKEND_PAIRS)
    @pytest.mark.parametrize(
        "matrix_dtype, blocks_dtype, expected_dtype",
        [
            ("float32", "float32", "float32"),
            ("float64", "float64", "float64"),
            ("float32", "float64", "float64"),
        ],
    )
    def test_product_dtype_is_the_promotion_of_the_matrix_and_the_blocks(
        self,
        matrix_backend: str,
        blocks_backend: str,
        matrix_dtype: str,
        blocks_dtype: str,
        expected_dtype: str,
        rng: np.random.Generator,
    ):
        matrix_values = rng.normal(size=(N_COLUMNS, N_COLUMNS)).astype(matrix_dtype)
        groups = [
            (indices, stacked.astype(blocks_dtype))
            for indices, stacked in self._random_groups(rng, STEPS_CASES["overlapping_steps"])
        ]

        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            self._to_backend(matrix_values, matrix_backend), self._groups_to_backend(groups, blocks_backend)
        )

        assert str(product.dtype).replace("torch.", "") == expected_dtype
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(matrix_values, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "matrix_backend, blocks_backend",
        [
            pytest.param("numpy", "numpy", id="numpy_matrix_numpy_blocks"),
            *[
                pytest.param(
                    matrix_backend,
                    blocks_backend,
                    id=f"{matrix_backend}_matrix_{blocks_backend}_blocks",
                )
                for matrix_backend, blocks_backend in [("torch", "torch"), ("torch", "numpy"), ("numpy", "torch")]
            ],
        ],
    )
    def test_single_precision_blocks_multiply_a_double_precision_matrix(
        self, matrix_backend: str, blocks_backend: str, rng: np.random.Generator
    ):
        matrix_values = rng.normal(size=(N_COLUMNS, N_COLUMNS))
        groups = [
            (indices, stacked.astype(np.float32))
            for indices, stacked in self._random_groups(rng, STEPS_CASES["overlapping_steps"])
        ]

        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            self._to_backend(matrix_values, matrix_backend), self._groups_to_backend(groups, blocks_backend)
        )

        assert str(product.dtype).replace("torch.", "") == "float64"
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(matrix_values, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("blocks_require_grad", [False, True])
    def test_product_without_gradient_has_no_graph(self, blocks_require_grad: bool, rng: np.random.Generator):
        matrix_values = rng.normal(size=(N_COLUMNS, N_COLUMNS))
        groups = self._random_groups(rng, STEPS_CASES["two_sizes"])
        torch_groups = [
            (torch.from_numpy(indices), torch.from_numpy(stacked).requires_grad_(blocks_require_grad))
            for indices, stacked in groups
        ]

        with torch.set_grad_enabled(not blocks_require_grad):
            product = SingleParticleTransitionMatrixOperation.multiply_groups(
                torch.from_numpy(matrix_values), torch_groups
            )

        assert not product.requires_grad
        assert product.grad_fn is None
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(matrix_values, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("steps_name", list(STEPS_CASES))
    @pytest.mark.parametrize("batch_size", [None, BATCH_SIZE])
    @pytest.mark.parametrize("orthogonal", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_gradients_are_those_of_the_dense_product(
        self,
        steps_name: str,
        batch_size: Optional[int],
        orthogonal: bool,
        dtype: torch.dtype,
        rng: np.random.Generator,
    ):
        matrix_values = rng.normal(size=(N_COLUMNS, N_COLUMNS))
        groups = self._random_groups(rng, STEPS_CASES[steps_name], batch_size=batch_size, orthogonal=orthogonal)
        weights = torch.from_numpy(rng.normal(size=(N_COLUMNS, N_COLUMNS))).to(dtype)
        gradients = []
        for product_function in (
            SingleParticleTransitionMatrixOperation.multiply_groups,
            self._torch_dense_product,
        ):
            matrix = torch.from_numpy(matrix_values.copy()).to(dtype).requires_grad_()
            torch_groups = [
                (torch.from_numpy(indices), torch.from_numpy(stacked.copy()).to(dtype).requires_grad_())
                for indices, stacked in groups
            ]
            product = product_function(matrix, torch_groups)
            assert product.requires_grad
            gradients.append(
                torch.autograd.grad(
                    torch.real((product * weights).sum()), [matrix, *[stacked for _, stacked in torch_groups]]
                )
            )

        for gradient, expected in zip(*gradients):
            torch.testing.assert_close(gradient, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize(
        "orthogonal, dtype, is_reversible",
        [
            (True, torch.float64, True),
            (True, torch.float32, True),
            (False, torch.float64, False),
            (True, torch.complex128, False),
        ],
    )
    def test_only_orthogonal_real_blocks_take_the_reversible_backward_pass(
        self, orthogonal: bool, dtype: torch.dtype, is_reversible: bool, rng: np.random.Generator
    ):
        groups = self._random_groups(rng, STEPS_CASES["overlapping_steps"], orthogonal=orthogonal)
        torch_groups = [
            (torch.from_numpy(indices), torch.from_numpy(stacked).to(dtype).requires_grad_())
            for indices, stacked in groups
        ]

        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            torch.eye(N_COLUMNS, dtype=dtype), torch_groups
        )

        assert product.requires_grad
        assert ("Reversible" in type(product.grad_fn).__name__) == is_reversible

    @pytest.mark.parametrize("batch_size", [None, BATCH_SIZE])
    @pytest.mark.parametrize("groups_backend", ["numpy", "torch"])
    def test_gradcheck_with_respect_to_the_matrix(
        self, batch_size: Optional[int], groups_backend: str, rng: np.random.Generator
    ):
        groups = self._groups_to_backend(
            self._random_groups(rng, STEPS_CASES["two_sizes"], batch_size=batch_size), groups_backend
        )
        matrix = torch.from_numpy(rng.normal(size=(N_COLUMNS, N_COLUMNS))).requires_grad_()

        assert torch.autograd.gradcheck(
            lambda matrix_input: SingleParticleTransitionMatrixOperation.multiply_groups(matrix_input, groups),
            (matrix,),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    @pytest.mark.parametrize("angles_shape", [(2, 3, 2), (2, 3, BATCH_SIZE, 2)])
    def test_gradcheck_with_respect_to_the_angles_of_rotations(self, angles_shape, rng: np.random.Generator):
        angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=angles_shape)).requires_grad_()
        indices = [np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), np.array([[2, 3, 4, 5], [6, 7, 8, 9]])]

        def product_of(angles_input: torch.Tensor) -> torch.Tensor:
            first = SptmCompRyRy.compute_sptm(angles_input[0])
            second = SptmCompRzRz.compute_sptm(angles_input[1, :2])
            if angles_input.ndim == 4:
                first, second = first.transpose(0, 1), second.transpose(0, 1)
            return SingleParticleTransitionMatrixOperation.multiply_groups(
                torch.eye(N_COLUMNS, dtype=torch.float64), [(indices[0], first), (indices[1], second)]
            )

        assert "Reversible" in type(product_of(angles).grad_fn).__name__
        assert torch.autograd.gradcheck(
            product_of, (angles,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    @pytest.mark.parametrize("batch_size", [None, BATCH_SIZE])
    @pytest.mark.parametrize("indices_backend", ["numpy", "torch"])
    def test_product_inside_a_functional_transform(
        self, batch_size: Optional[int], indices_backend: str, rng: np.random.Generator
    ):
        angles_shape = (2, 2, 2) if batch_size is None else (2, 2, batch_size, 2)
        angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=angles_shape))
        indices = [
            self._to_backend(np.array([[0, 1, 2, 3], [8, 9, 10, 11]]), indices_backend),
            self._to_backend(np.array([[2, 3, 4, 5], [6, 7, 8, 9]]), indices_backend),
        ]

        def product_of(angles_input: torch.Tensor) -> torch.Tensor:
            first = SptmCompRyRy.compute_sptm(angles_input[0])
            second = SptmCompRzRz.compute_sptm(angles_input[1])
            if angles_input.ndim == 4:
                first, second = first.transpose(0, 1), second.transpose(0, 1)
            return SingleParticleTransitionMatrixOperation.multiply_groups(
                torch.eye(N_COLUMNS, dtype=torch.float64), [(indices[0], first), (indices[1], second)]
            )

        torch.testing.assert_close(
            torch.func.jacrev(product_of)(angles),
            torch.autograd.functional.jacobian(product_of, angles),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
