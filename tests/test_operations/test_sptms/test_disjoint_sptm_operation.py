from typing import Any, List, Optional, Sequence

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane import numpy as pnp
from pennylane.wires import Wires

from matchcake.operations import SptmCompRyRy, SptmCompRzRz, SptmCompZX
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
BACKENDS = ("numpy", "torch", "autograd")
# wires of every block, in the order the blocks are given
BLOCKS_WIRES_CASES = {
    "two_adjacent": [[0, 1], [2, 3]],
    "two_with_a_gap": [[0, 1], [3, 4]],
    "three_blocks": [[0, 1], [2, 3], [6, 7]],
    "different_sizes": [[0, 1, 2], [4, 5]],
    "unsorted": [[4, 5], [0, 1], [2, 3]],
    "unsorted_different_sizes": [[5, 6], [1, 2, 3]],
    "not_starting_at_zero": [[2, 3], [5, 6]],
}
# which blocks are batched, None for every block
BATCHED_BLOCKS_CASES = [
    pytest.param((), id="unbatched"),
    pytest.param(None, id="batched"),
    pytest.param((0,), id="first_batched"),
    pytest.param((1,), id="second_batched"),
]


class TestDisjointSptmOperation:
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
    def _random_matrices(
        rng: np.random.Generator,
        blocks_wires: List[List[int]],
        batched_blocks: Optional[Sequence[int]] = (),
    ) -> List[np.ndarray]:
        matrices = []
        for index, wires in enumerate(blocks_wires):
            is_batched = batched_blocks is None or index in batched_blocks
            size = 2 * len(wires)
            matrices.append(rng.normal(size=(BATCH_SIZE, size, size) if is_batched else (size, size)))
        return matrices

    @classmethod
    def _make_blocks(
        cls,
        matrices: List[np.ndarray],
        blocks_wires: List[List[int]],
        backend: str,
    ) -> List[SingleParticleTransitionMatrixOperation]:
        return [
            SingleParticleTransitionMatrixOperation(cls._to_backend(matrix, backend), wires=wires)
            for matrix, wires in zip(matrices, blocks_wires)
        ]

    @classmethod
    def _block_diagonal(
        cls,
        matrices: Sequence[Any],
        blocks_wires: List[List[int]],
        all_wires: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        block_wires = [wire for wires in blocks_wires for wire in wires]
        if all_wires is None:
            all_wires = list(range(min(block_wires), max(block_wires) + 1))
        size = 2 * len(all_wires)
        arrays = [cls._to_numpy(matrix) for matrix in matrices]
        batch_shapes = [array.shape[:-2] for array in arrays if array.ndim > 2]
        batch_shape = batch_shapes[0] if batch_shapes else ()
        dense = np.zeros((*batch_shape, size, size))
        dense[...] = np.eye(size)
        for array, wires in zip(arrays, blocks_wires):
            start = 2 * list(all_wires).index(min(wires))
            dense[..., start : start + array.shape[-2], start : start + array.shape[-1]] = array
        return dense

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", list(BLOCKS_WIRES_CASES))
    @pytest.mark.parametrize("batched_blocks", BATCHED_BLOCKS_CASES)
    def test_dense_matrix_is_the_block_diagonal_embedding(self, backend, case_name, batched_blocks, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        matrices = self._random_matrices(rng, blocks_wires, batched_blocks)
        expected = self._block_diagonal(matrices, blocks_wires)

        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))
        dense = operation.sptm_matrix()

        expected_type = {"numpy": np.ndarray, "torch": torch.Tensor, "autograd": pnp.tensor}[backend]
        assert isinstance(dense, expected_type)
        assert tuple(dense.shape) == expected.shape
        np.testing.assert_allclose(
            self._to_numpy(dense), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            self._to_numpy(operation.single_particle_transition_matrix),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", list(BLOCKS_WIRES_CASES))
    def test_wires_are_the_continuous_span_of_the_blocks(self, backend, case_name, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        all_wires = [wire for wires in blocks_wires for wire in wires]
        matrices = self._random_matrices(rng, blocks_wires)

        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))

        expected_wires = Wires(range(min(all_wires), max(all_wires) + 1))
        assert operation.wires == expected_wires
        assert operation.cs_wires == expected_wires
        assert tuple(operation.shape) == (2 * len(expected_wires), 2 * len(expected_wires))

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", list(BLOCKS_WIRES_CASES))
    def test_blocks_are_kept_sorted_by_wires(self, backend, case_name, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        matrices = self._random_matrices(rng, blocks_wires)
        blocks = self._make_blocks(matrices, blocks_wires, backend)

        operation = DisjointSptmOperation(blocks)

        kept_blocks = operation.blocks
        assert kept_blocks is not None
        assert operation.num_params == len(blocks) == len(operation.data)
        assert [block.wires.tolist() for block in kept_blocks] == sorted(blocks_wires)
        assert {id(block) for block in kept_blocks} == {id(block) for block in blocks}
        for block, parameter in zip(kept_blocks, operation.data):
            assert parameter is block.sptm_matrix()

    def test_blocks_returns_a_copy_of_the_list(self, rng):
        blocks_wires = BLOCKS_WIRES_CASES["three_blocks"]
        operation = DisjointSptmOperation(
            self._make_blocks(self._random_matrices(rng, blocks_wires), blocks_wires, "numpy")
        )
        blocks = operation.blocks
        assert blocks is not None

        blocks.pop()

        kept_blocks = operation.blocks
        assert kept_blocks is not None
        assert len(kept_blocks) == len(blocks_wires)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize(
        "batched_blocks, expected_batch_size",
        [
            pytest.param((), None, id="unbatched"),
            pytest.param(None, BATCH_SIZE, id="batched"),
            pytest.param((0,), BATCH_SIZE, id="first_batched"),
            pytest.param((2,), BATCH_SIZE, id="last_batched"),
        ],
    )
    def test_batch_size(self, backend, batched_blocks, expected_batch_size, rng):
        blocks_wires = BLOCKS_WIRES_CASES["three_blocks"]
        matrices = self._random_matrices(rng, blocks_wires, batched_blocks)

        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))

        assert operation.batch_size == expected_batch_size
        assert operation._dense_matrix is None
        expected_shape = (16, 16) if expected_batch_size is None else (expected_batch_size, 16, 16)
        assert tuple(operation.sptm_matrix().shape) == expected_shape

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", list(BLOCKS_WIRES_CASES))
    def test_column_blocks(self, backend, case_name, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        all_wires = Wires(range(0, 10))
        matrices = self._random_matrices(rng, blocks_wires)
        blocks = self._make_blocks(matrices, blocks_wires, backend)

        operation = DisjointSptmOperation(blocks)
        column_blocks = operation.column_blocks(all_wires)

        expected = sorted(zip([2 * min(wires) for wires in blocks_wires], range(len(blocks))))
        assert [start for start, _ in column_blocks] == [start for start, _ in expected]
        for (_, matrix), (_, block_index) in zip(column_blocks, expected):
            assert matrix is blocks[block_index].sptm_matrix()
        assert operation._dense_matrix is None

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_dense_matrix_is_built_lazily_and_only_once(self, backend, rng):
        blocks_wires = BLOCKS_WIRES_CASES["two_with_a_gap"]
        matrices = self._random_matrices(rng, blocks_wires)

        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))

        assert operation._dense_matrix is None
        assert operation.wires == Wires(range(5))
        assert operation.batch_size is None
        assert operation.num_params == 2
        assert operation.blocks is not None
        operation.column_blocks(Wires(range(6)))
        assert operation._dense_matrix is None
        dense = operation.sptm_matrix()
        assert operation._dense_matrix is dense
        assert operation.sptm_matrix() is dense

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_dense_matrix_does_not_share_memory_with_the_blocks(self, backend, rng):
        blocks_wires = BLOCKS_WIRES_CASES["two_adjacent"]
        matrices = self._random_matrices(rng, blocks_wires)
        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))

        dense = operation.sptm_matrix()
        dense[...] = 0

        kept_blocks = operation.blocks
        assert kept_blocks is not None
        for block, matrix in zip(kept_blocks, matrices):
            np.testing.assert_array_equal(self._to_numpy(block.sptm_matrix()), matrix)

    def test_integer_blocks_give_an_integer_dense_matrix(self):
        operation = DisjointSptmOperation([SptmCompZX(wires=[0, 1]), SptmCompZX(wires=[3, 4])])
        swap_matrix = SptmCompZX(wires=[0, 1]).sptm_matrix()

        dense = operation.sptm_matrix()

        assert np.issubdtype(dense.dtype, np.integer)
        np.testing.assert_array_equal(dense, self._block_diagonal([swap_matrix, swap_matrix], [[0, 1], [3, 4]]))

    def test_integer_and_float_blocks_give_a_float_dense_matrix(self):
        rotation = SptmCompRyRy(np.array([0.3, 0.4]), wires=[2, 3])
        swap = SptmCompZX(wires=[0, 1])

        dense = DisjointSptmOperation([swap, rotation]).sptm_matrix()

        assert np.issubdtype(dense.dtype, np.floating)
        np.testing.assert_allclose(
            dense,
            self._block_diagonal([swap.sptm_matrix(), rotation.sptm_matrix()], [[0, 1], [2, 3]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "backends",
        [
            pytest.param(("numpy", "torch"), id="numpy_and_torch"),
            pytest.param(("torch", "numpy"), id="torch_and_numpy"),
            pytest.param(("numpy", "autograd"), id="numpy_and_autograd"),
            pytest.param(("autograd", "numpy"), id="autograd_and_numpy"),
        ],
    )
    @pytest.mark.parametrize("batched_blocks", BATCHED_BLOCKS_CASES)
    def test_blocks_of_different_backends(self, backends, batched_blocks, rng):
        blocks_wires = BLOCKS_WIRES_CASES["different_sizes"]
        matrices = self._random_matrices(rng, blocks_wires, batched_blocks)
        blocks = [
            SingleParticleTransitionMatrixOperation(self._to_backend(matrix, backend), wires=wires)
            for matrix, wires, backend in zip(matrices, blocks_wires, backends)
        ]

        dense = DisjointSptmOperation(blocks).sptm_matrix()

        assert isinstance(dense, torch.Tensor if "torch" in backends else pnp.tensor)
        np.testing.assert_allclose(
            self._to_numpy(dense),
            self._block_diagonal(matrices, blocks_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_integer_block_before_a_pennylane_numpy_block_gives_a_float_dense_matrix(self):
        swap = SptmCompZX(wires=[0, 1])
        rotation = SptmCompRyRy(pnp.array([0.3, 0.4], requires_grad=True), wires=[2, 3])

        dense = DisjointSptmOperation([swap, rotation]).sptm_matrix()

        np.testing.assert_allclose(
            self._to_numpy(dense),
            self._block_diagonal([swap.sptm_matrix(), rotation.sptm_matrix()], [[0, 1], [2, 3]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_pennylane_numpy_block_before_an_integer_block_gives_a_float_dense_matrix(self):
        rotation = SptmCompRyRy(pnp.array([0.3, 0.4], requires_grad=True), wires=[0, 1])
        swap = SptmCompZX(wires=[2, 3])

        dense = DisjointSptmOperation([swap, rotation]).sptm_matrix()

        np.testing.assert_allclose(
            self._to_numpy(dense),
            self._block_diagonal([rotation.sptm_matrix(), swap.sptm_matrix()], [[0, 1], [2, 3]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_no_blocks_raises(self):
        with pytest.raises(ValueError, match="without blocks"):
            DisjointSptmOperation([])

    @pytest.mark.parametrize(
        "blocks_wires",
        [
            pytest.param([[0, 1], [1, 2]], id="one_shared_wire"),
            pytest.param([[0, 1], [0, 1]], id="same_wires"),
            pytest.param([[0, 1, 2, 3], [1, 2]], id="nested"),
            pytest.param([[0, 1], [3, 4], [4, 5]], id="third_overlaps_second"),
            pytest.param([[0, 3], [1, 2]], id="within_the_continuous_span"),
        ],
    )
    def test_overlapping_blocks_raise(self, blocks_wires, rng):
        matrices = [rng.normal(size=(2 * (max(wires) - min(wires) + 1),) * 2) for wires in blocks_wires]
        blocks = self._make_blocks(matrices, blocks_wires, "numpy")

        with pytest.raises(ValueError, match="disjoint wires"):
            DisjointSptmOperation(blocks)

    def test_from_operations_without_operations_is_none(self):
        assert SingleParticleTransitionMatrixOperation.from_operations([]) is None

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_from_operations_with_one_operation_is_that_operation(self, backend, rng):
        blocks = self._make_blocks(self._random_matrices(rng, [[1, 2]]), [[1, 2]], backend)

        operation = SingleParticleTransitionMatrixOperation.from_operations(blocks)

        assert operation is blocks[0]

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", list(BLOCKS_WIRES_CASES))
    def test_from_operations_with_several_operations_is_a_disjoint_operation(self, backend, case_name, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        matrices = self._random_matrices(rng, blocks_wires)

        operation = SingleParticleTransitionMatrixOperation.from_operations(
            iter(self._make_blocks(matrices, blocks_wires, backend))
        )

        assert isinstance(operation, DisjointSptmOperation)
        assert operation._dense_matrix is None
        np.testing.assert_allclose(
            self._to_numpy(operation.sptm_matrix()),
            self._block_diagonal(matrices, blocks_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_from_operations_called_on_a_subclass_is_a_disjoint_operation(self):
        rotations = [
            SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]),
            SptmCompRzRz(np.array([0.5, 0.6]), wires=[2, 3]),
        ]

        operation = SptmCompRyRy.from_operations(rotations)

        assert isinstance(operation, DisjointSptmOperation)
        np.testing.assert_allclose(
            operation.sptm_matrix(),
            self._block_diagonal([rotation.sptm_matrix() for rotation in rotations], [[0, 1], [2, 3]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_hyperparameters_default_and_overrides(self, rng):
        blocks_wires = BLOCKS_WIRES_CASES["two_adjacent"]
        blocks = self._make_blocks(self._random_matrices(rng, blocks_wires), blocks_wires, "numpy")

        default_operation = DisjointSptmOperation(blocks)
        operation = SingleParticleTransitionMatrixOperation.from_operations(
            blocks, check_real=False, clip_angles=True, normalize=True, extra_option=1
        )

        assert default_operation.hyperparameters == {
            "clip_angles": SingleParticleTransitionMatrixOperation.DEFAULT_CLIP_ANGLES,
            "check_angles": SingleParticleTransitionMatrixOperation.DEFAULT_CHECK_ANGLES,
            "check_matrix": SingleParticleTransitionMatrixOperation.DEFAULT_CHECK_MATRIX,
            "check_real": SingleParticleTransitionMatrixOperation.DEFAULT_CHECK_REAL,
        }
        assert isinstance(operation, DisjointSptmOperation)
        assert operation.hyperparameters["check_real"] is False
        assert operation.hyperparameters["clip_angles"] is True
        assert operation.kwargs == {"extra_option": 1}

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", ["two_with_a_gap", "unsorted_different_sizes", "not_starting_at_zero"])
    @pytest.mark.parametrize("batched_blocks", BATCHED_BLOCKS_CASES)
    def test_pad_and_sptm_matrix_with_wire_order(self, backend, case_name, batched_blocks, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        all_wires = list(range(0, 9))
        matrices = self._random_matrices(rng, blocks_wires, batched_blocks)
        expected = self._block_diagonal(matrices, blocks_wires, all_wires)
        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))

        padded = operation.pad(Wires(all_wires))

        assert padded.wires == Wires(all_wires)
        np.testing.assert_allclose(
            self._to_numpy(padded.sptm_matrix()), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            self._to_numpy(operation.sptm_matrix(all_wires)),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._to_numpy(operation.sptm_matrix()),
            self._block_diagonal(matrices, blocks_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_pad_to_its_own_wires_is_itself(self, backend, rng):
        blocks_wires = BLOCKS_WIRES_CASES["two_with_a_gap"]
        operation = DisjointSptmOperation(
            self._make_blocks(self._random_matrices(rng, blocks_wires), blocks_wires, backend)
        )

        assert operation.pad(operation.wires) is operation

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("case_name", ["two_with_a_gap", "unsorted_different_sizes"])
    @pytest.mark.parametrize("batched_blocks", BATCHED_BLOCKS_CASES)
    def test_adjoint(self, backend, case_name, batched_blocks, rng):
        blocks_wires = BLOCKS_WIRES_CASES[case_name]
        matrices = self._random_matrices(rng, blocks_wires, batched_blocks)
        operation = DisjointSptmOperation(self._make_blocks(matrices, blocks_wires, backend))

        adjoint = operation.adjoint()

        assert adjoint.wires == operation.wires
        np.testing.assert_allclose(
            self._to_numpy(adjoint.sptm_matrix()),
            self._block_diagonal([np.swapaxes(matrix, -1, -2) for matrix in matrices], blocks_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._to_numpy(operation.sptm_matrix()),
            self._block_diagonal(matrices, blocks_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("batched_blocks", BATCHED_BLOCKS_CASES)
    def test_dense_matrix_gradcheck(self, batched_blocks, rng):
        blocks_wires = BLOCKS_WIRES_CASES["two_with_a_gap"]
        matrices = tuple(
            torch.from_numpy(matrix).requires_grad_()
            for matrix in self._random_matrices(rng, blocks_wires, batched_blocks)
        )

        def dense_matrix(*block_matrices: torch.Tensor) -> torch.Tensor:
            blocks = [
                SingleParticleTransitionMatrixOperation(matrix, wires=wires)
                for matrix, wires in zip(block_matrices, blocks_wires)
            ]
            return DisjointSptmOperation(blocks).sptm_matrix()

        assert torch.autograd.gradcheck(
            dense_matrix,
            matrices,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    def test_decomposition_is_the_blocks(self):
        blocks = [SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]), SptmCompRzRz(np.array([0.5, 0.1]), wires=[3, 4])]
        operation = DisjointSptmOperation(blocks)
        decomposition = operation.decomposition()
        assert [block.wires for block in decomposition] == [block.wires for block in blocks]
        assert all(new is old for new, old in zip(decomposition, blocks))

    def test_decomposition_is_queued_when_recording(self):
        blocks = [SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]), SptmCompRzRz(np.array([0.5, 0.1]), wires=[3, 4])]
        operation = DisjointSptmOperation(blocks)
        with qml.queuing.AnnotatedQueue() as queue:
            operation.decomposition()
        assert len(queue.queue) == len(blocks)

    def test_qubit_matrix_of_the_decomposition_matches_the_dense_operation(self):
        blocks = [SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]), SptmCompRzRz(np.array([0.5, 0.1]), wires=[2, 3])]
        operation = DisjointSptmOperation(blocks)
        dense = SingleParticleTransitionMatrixOperation(operation.sptm_matrix(), wires=operation.wires)
        np.testing.assert_allclose(
            qml.matrix(operation, wire_order=range(4)),
            qml.matrix(dense, wire_order=range(4)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @staticmethod
    def _three_rotations(backend: str = "numpy") -> list:
        angles: list = [np.array([0.3, 0.4]), np.array([0.5, 0.1]), np.array([0.2, 0.7])]
        if backend == "torch":
            angles = [torch.from_numpy(angle) for angle in angles]
        return [SptmCompRyRy(angle, wires=[2 * index, 2 * index + 1]) for index, angle in enumerate(angles)]

    def test_a_dense_operation_equals_the_same_operation_kept_as_blocks(self):
        operation = DisjointSptmOperation(self._three_rotations())
        dense = SingleParticleTransitionMatrixOperation(operation.sptm_matrix(), wires=operation.wires)
        assert qml.equal(dense, operation)

    def test_two_splits_of_the_same_operation_are_equal(self):
        first, second, third = self._three_rotations()
        nested = DisjointSptmOperation([DisjointSptmOperation([first, second]), third])
        merged = SingleParticleTransitionMatrixOperation.from_operations([first, second])
        merged = SingleParticleTransitionMatrixOperation(merged.sptm_matrix(), wires=merged.wires)
        assert qml.equal(DisjointSptmOperation([first, second, third]), nested)
        assert qml.equal(DisjointSptmOperation([first, second, third]), DisjointSptmOperation([merged, third]))

    def test_equal_is_false_with_the_block_form_first_as_pennylane_checks_the_type(self):
        operation = DisjointSptmOperation(self._three_rotations())
        dense = SingleParticleTransitionMatrixOperation(operation.sptm_matrix(), wires=operation.wires)
        assert not qml.equal(operation, dense)

    @pytest.mark.parametrize(
        "make_other",
        [
            lambda dense: SingleParticleTransitionMatrixOperation(dense.sptm_matrix().T, wires=dense.wires),
            lambda dense: SingleParticleTransitionMatrixOperation(dense.sptm_matrix(), wires=range(1, 7)),
            lambda dense: SingleParticleTransitionMatrixOperation(
                dense.sptm_matrix(), wires=dense.wires, check_angles=True
            ),
            lambda dense: SingleParticleTransitionMatrixOperation(
                np.stack([dense.sptm_matrix()] * 2), wires=dense.wires
            ),
            lambda dense: SingleParticleTransitionMatrixOperation(
                torch.from_numpy(dense.sptm_matrix()), wires=dense.wires
            ),
            lambda dense: SingleParticleTransitionMatrixOperation(
                pnp.array(dense.sptm_matrix(), requires_grad=True), wires=dense.wires
            ),
        ],
    )
    def test_a_dense_operation_that_differs_is_not_equal(self, make_other):
        operation = DisjointSptmOperation(self._three_rotations())
        dense = SingleParticleTransitionMatrixOperation(operation.sptm_matrix(), wires=operation.wires)
        assert not qml.equal(make_other(dense), operation)

    def test_interface_and_trainability_checks_can_be_skipped(self):
        operation = DisjointSptmOperation(self._three_rotations())
        dense = SingleParticleTransitionMatrixOperation(
            torch.from_numpy(operation.sptm_matrix()).requires_grad_(), wires=operation.wires
        )
        assert qml.equal(dense, operation, check_interface=False, check_trainability=False)
        assert not qml.equal(dense, operation, check_interface=False)

    def test_two_plain_operations_are_compared_as_before(self):
        first, second, _ = self._three_rotations()
        same = SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1])
        assert qml.equal(first, same)
        assert not qml.equal(first, second)

    def test_the_dense_matrix_can_be_replaced(self):
        operation = DisjointSptmOperation(self._three_rotations())
        replacement = np.eye(12)
        assert operation._stored_matrix is None and operation._params is None

        operation._matrix = replacement

        assert operation.sptm_matrix() is replacement
        assert operation._stored_matrix is None
        assert operation.batch_size is None

    @pytest.mark.parametrize("batch_sizes", [(1, 3), (3, 1)])
    def test_pennylane_numpy_blocks_with_a_batch_of_one_are_broadcast_to_the_largest_batch(self, batch_sizes):
        """The dense fallback used to take the batch of the first batched block, whatever the others."""
        generator = np.random.default_rng(TEST_SEED)
        angles = [generator.uniform(0.0, 2.0, size=(batch_size, 2)) for batch_size in batch_sizes]
        blocks = [
            SptmCompRzRz(pnp.array(angles[0], requires_grad=True), wires=[0, 1]),
            SptmCompRzRz(pnp.array(angles[1], requires_grad=True), wires=[2, 3]),
        ]
        operation = DisjointSptmOperation(blocks)
        expected = np.zeros((3, 8, 8))
        expected[:, :4, :4] = np.asarray(SptmCompRzRz(angles[0], wires=[0, 1]).sptm_matrix())
        expected[:, 4:, 4:] = np.asarray(SptmCompRzRz(angles[1], wires=[2, 3]).sptm_matrix())
        assert operation.batch_size == 3
        np.testing.assert_allclose(
            np.asarray(operation.sptm_matrix()), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
