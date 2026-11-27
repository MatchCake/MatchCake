from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from pennylane import numpy as pnp

from matchcake.devices.contraction_strategies import (
    BulkContractionStrategy,
    contraction_strategy_map,
    get_contraction_strategy,
)
from matchcake.devices.contraction_strategies.contraction_container import _ContractionMatchgatesContainer
from matchcake.devices.sptm_circuit_records import SptmCircuitRecords
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
)

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

N_COLUMNS = 16
BATCH_SIZE = 3
N_ENTRIES = 30
ROTATIONS = {"rxrx": SptmCompRxRx, "ryry": SptmCompRyRy, "rzrz": SptmCompRzRz}
ENTRY_KINDS = ("rxrx", "ryry", "rzrz", "zx", "block4", "block6")
# (first columns, block sizes, number of columns, expected layers)
SCHEDULE_CASES = {
    "no_block": ([], [], 8, []),
    "one_block": ([2], [4], 8, [0]),
    "disjoint_blocks_share_a_layer": ([0, 4, 8], [4, 4, 4], 12, [0, 0, 0]),
    "same_columns_are_ordered": ([0, 0, 0], [4, 4, 4], 4, [0, 1, 2]),
    "overlapping_blocks_are_ordered": ([0, 2, 4], [4, 4, 4], 8, [0, 1, 2]),
    "brick_wall": ([0, 4, 2, 0, 4], [4, 4, 4, 4, 4], 8, [0, 0, 1, 2, 2]),
    "a_late_disjoint_block_takes_the_first_layer": ([0, 0, 0, 4], [4, 4, 4, 4], 8, [0, 1, 2, 0]),
    "a_block_waits_for_the_deepest_of_its_columns": ([0, 0, 4, 2], [4, 4, 4, 4], 8, [0, 1, 0, 2]),
    "disjoint_blocks_of_two_sizes": ([0, 6], [6, 4], 10, [0, 0]),
    "overlapping_blocks_of_two_sizes": ([0, 4, 2, 0], [4, 4, 4, 6], 8, [0, 0, 1, 2]),
    "a_large_block_holds_back_the_blocks_after_it": ([0, 0, 4, 0, 6], [4, 4, 4, 8, 4], 10, [0, 1, 0, 2, 3]),
}


class TestBulkContractionStrategy:
    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _random_entries(
        rng: np.random.Generator, backend: str, batching: str, n_entries: int = N_ENTRIES
    ) -> List[Tuple[int, int, Optional[type], Any]]:
        entries: List[Tuple[int, int, Optional[type], Any]] = []
        for index in range(n_entries):
            kind = ENTRY_KINDS[int(rng.integers(len(ENTRY_KINDS)))]
            size = 6 if kind == "block6" else 4
            start = 2 * int(rng.integers((N_COLUMNS - size) // 2 + 1))
            is_batched = batching == "batched" or (batching == "mixed" and bool(rng.integers(2)))
            batch_shape = (BATCH_SIZE,) if is_batched else ()
            use_torch = backend == "torch" or (backend == "mixed" and index % 2 == 1)
            tensor: Any
            if kind == "zx":
                entries.append((start, size, SptmCompZX, np.zeros(0)))
                continue
            if kind in ROTATIONS:
                tensor = rng.uniform(-np.pi, np.pi, size=(*batch_shape, 2))
            else:
                tensor, _ = np.linalg.qr(rng.normal(size=(*batch_shape, size, size)))
            entries.append((start, size, ROTATIONS.get(kind), torch.from_numpy(tensor) if use_torch else tensor))
        return entries

    @staticmethod
    def _record(entries: Sequence[Tuple[Any, ...]]) -> SptmCircuitRecords:
        records = SptmCircuitRecords()
        for start, size, gate_class, tensor in entries:
            if gate_class is None:
                records.add_blocks([(start, tensor)])
            else:
                records.add_gate(gate_class, start, size, tensor)
        return records

    @classmethod
    def _dense_product(cls, entries: Sequence[Tuple[Any, ...]]) -> np.ndarray:
        result = np.eye(N_COLUMNS)
        for start, size, gate_class, tensor in entries:
            block = cls._to_numpy(tensor)
            if gate_class is not None:
                block = gate_class.compute_sptm(block)
            padded = np.zeros((*block.shape[:-2], N_COLUMNS, N_COLUMNS))
            padded[...] = np.eye(N_COLUMNS)
            padded[..., start : start + size, start : start + size] = block
            result = result @ padded
        return result

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.fixture
    def strategy(self) -> BulkContractionStrategy:
        return BulkContractionStrategy()

    def test_strategy_is_registered_under_its_name(self):
        assert BulkContractionStrategy.NAME == "Bulk"
        assert contraction_strategy_map["bulk"] is BulkContractionStrategy
        assert isinstance(get_contraction_strategy("bulk"), BulkContractionStrategy)
        assert isinstance(get_contraction_strategy(" Bulk "), BulkContractionStrategy)

    def test_operations_pass_through_untouched(self, strategy: BulkContractionStrategy):
        operations = [
            SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]),
            SptmCompRzRz(np.array([0.5, 0.1]), wires=[0, 1]),
            SptmCompZX(wires=[1, 2]),
        ]

        assert strategy(operations) is operations
        assert strategy(operations, show_progress=True) is operations
        for operation in operations:
            next_operations = strategy.get_next_operations(operation)
            assert len(next_operations) == 1
            assert next_operations[0] is operation
            assert operation._stored_matrix is None or isinstance(operation, SptmCompZX)
        assert strategy.get_reminding() is None
        assert len(strategy.container) == 0

    def test_container_is_empty_and_follows_the_dtype_name(self, strategy: BulkContractionStrategy):
        assert type(strategy.get_container()) is _ContractionMatchgatesContainer
        assert strategy.dtype_name is None

        strategy.dtype_name = "float32"
        strategy.reset()

        assert strategy.dtype_name == "float32"
        assert strategy.container.dtype_name == "float32"

    @pytest.mark.parametrize("case_name", list(SCHEDULE_CASES))
    def test_schedule_gives_each_block_its_earliest_layer(self, case_name: str):
        starts, sizes, n_columns, expected = SCHEDULE_CASES[case_name]

        layers = BulkContractionStrategy._schedule(
            np.array(starts, dtype=np.int64), np.array(sizes, dtype=np.int64), n_columns
        )

        assert layers.dtype == np.int64
        np.testing.assert_array_equal(layers, np.array(expected, dtype=np.int64))

    def test_schedule_of_random_blocks_respects_the_order_on_shared_columns(self, rng: np.random.Generator):
        entries = self._random_entries(rng, "numpy", "unbatched", n_entries=60)
        starts = np.array([start for start, _, _, _ in entries])
        sizes = np.array([size for _, size, _, _ in entries])

        layers = BulkContractionStrategy._schedule(starts, sizes, N_COLUMNS)

        for index in range(len(entries)):
            earlier = [
                layers[other]
                for other in range(index)
                if starts[other] < starts[index] + sizes[index] and starts[index] < starts[other] + sizes[other]
            ]
            assert layers[index] == (max(earlier) + 1 if earlier else 0)

    def test_plan_has_one_stack_per_layer_and_block_size(
        self, strategy: BulkContractionStrategy, rng: np.random.Generator
    ):
        angles = rng.uniform(-np.pi, np.pi, size=(4, 2))
        blocks = [np.linalg.qr(rng.normal(size=(6, 6)))[0], np.linalg.qr(rng.normal(size=(4, 4)))[0]]
        entries = [
            (0, 4, SptmCompRyRy, angles[0]),
            (4, 6, None, blocks[0]),
            (10, 4, SptmCompRzRz, angles[1]),
            (12, 4, SptmCompZX, np.zeros(0)),
            (0, 4, None, blocks[1]),
            (2, 4, SptmCompRyRy, angles[2]),
            (6, 4, SptmCompRyRy, angles[3]),
        ]

        groups = strategy.build_plan(self._record(entries), N_COLUMNS, "float64")

        assert groups is not None
        # within a stack the blocks come gate class by gate class, then explicit blocks
        assert [indices[:, 0].tolist() for indices, _ in groups] == [[0, 10], [4], [6, 12, 0], [2]]
        assert [tuple(indices.shape) for indices, _ in groups] == [(2, 4), (1, 6), (3, 4), (1, 4)]
        assert [tuple(stacked.shape) for _, stacked in groups] == [(2, 4, 4), (1, 6, 6), (3, 4, 4), (1, 4, 4)]
        for indices, _ in groups:
            np.testing.assert_array_equal(indices, indices[:, :1] + np.arange(indices.shape[1])[None, :])
        expected_stacks = [
            [SptmCompRyRy.compute_sptm(angles[0]), SptmCompRzRz.compute_sptm(angles[1])],
            [blocks[0]],
            [SptmCompRyRy.compute_sptm(angles[3]), SptmCompZX.compute_sptm(np.zeros(0)), blocks[1]],
            [SptmCompRyRy.compute_sptm(angles[2])],
        ]
        for (_, stacked), expected in zip(groups, expected_stacks):
            np.testing.assert_allclose(
                stacked, np.stack(expected), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    @pytest.mark.parametrize("backend", ["numpy", "torch", "mixed"])
    @pytest.mark.parametrize("batching", ["unbatched", "batched", "mixed"])
    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    @pytest.mark.parametrize("seed_offset", [0, 1])
    def test_plan_of_a_random_circuit_gives_the_ordered_product(
        self, strategy: BulkContractionStrategy, backend: str, batching: str, dtype_name: str, seed_offset: int
    ):
        entries = self._random_entries(np.random.default_rng(TEST_SEED + seed_offset), backend, batching)
        is_torch = any(isinstance(tensor, torch.Tensor) for _, _, _, tensor in entries)
        is_batched = any(tensor.ndim > (1 if gate_class is not None else 2) for _, _, gate_class, tensor in entries)

        groups = strategy.build_plan(self._record(entries), N_COLUMNS, dtype_name)

        assert groups is not None
        assert sum(indices.shape[0] for indices, _ in groups) == len(entries)
        for indices, stacked in groups:
            assert isinstance(indices, torch.Tensor if is_torch else np.ndarray)
            assert isinstance(stacked, torch.Tensor if is_torch else np.ndarray)
            assert str(stacked.dtype).replace("torch.", "") == dtype_name
            n_blocks, size = tuple(indices.shape)
            # blocks with and without batch are stacked apart, so that a batched gate does not make
            # every other block as large as the batch
            assert tuple(stacked.shape)[-3:] == (n_blocks, size, size)
            assert stacked.ndim == 3 or (is_batched and tuple(stacked.shape)[0] == BATCH_SIZE)
            columns = self._to_numpy(indices).reshape(-1)
            assert len(set(columns.tolist())) == columns.size
        product = SingleParticleTransitionMatrixOperation.multiply_groups(np.eye(N_COLUMNS, dtype=dtype_name), groups)
        expected = self._dense_product(entries)
        assert tuple(product.shape) == expected.shape
        assert str(product.dtype).replace("torch.", "") == dtype_name
        np.testing.assert_allclose(
            self._to_numpy(product), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_plan_does_not_change_the_recorded_tensors(
        self, strategy: BulkContractionStrategy, backend: str, rng: np.random.Generator
    ):
        entries = self._random_entries(rng, backend, "mixed")
        values = [self._to_numpy(tensor).copy() for _, _, _, tensor in entries]
        records = self._record(entries)

        groups = strategy.build_plan(records, N_COLUMNS, "float64")
        assert groups is not None
        product = SingleParticleTransitionMatrixOperation.multiply_groups(np.eye(N_COLUMNS), groups)
        product[...] = 0.0

        assert len(records) == len(entries)
        for (_, _, _, tensor), expected in zip(records.entries, values):
            np.testing.assert_array_equal(self._to_numpy(tensor), expected)

    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_torch_tensors_of_several_dtypes_are_stacked_together(
        self, strategy: BulkContractionStrategy, dtype_name: str, rng: np.random.Generator
    ):
        angles = rng.uniform(-np.pi, np.pi, size=(4, 2))
        block = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        entries = [
            (0, 4, SptmCompRyRy, torch.from_numpy(angles[0].astype(np.float32))),
            (4, 4, SptmCompRyRy, torch.from_numpy(angles[1])),
            (8, 4, SptmCompRyRy, angles[2]),
            (2, 4, None, torch.from_numpy(block.astype(np.float32))),
            (6, 4, None, torch.from_numpy(block)),
            (10, 4, None, np.eye(4, dtype=np.int64)),
            (12, 4, SptmCompRzRz, angles[3].astype(np.float32)),
        ]

        groups = strategy.build_plan(self._record(entries), N_COLUMNS, dtype_name)

        assert groups is not None
        assert all(stacked.dtype == getattr(torch, dtype_name) for _, stacked in groups)
        product = SingleParticleTransitionMatrixOperation.multiply_groups(torch.eye(N_COLUMNS), groups)
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(entries),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("as_gate", [True, False])
    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_no_plan_for_a_pennylane_numpy_tensor(
        self, strategy: BulkContractionStrategy, as_gate: bool, position: int, rng: np.random.Generator
    ):
        entries = [
            (0, 4, SptmCompRyRy, rng.uniform(size=2)),
            (4, 4, None, torch.eye(4, dtype=torch.float64)),
        ]
        if as_gate:
            entries.insert(position, (2, 4, SptmCompRzRz, pnp.array(rng.uniform(size=2), requires_grad=True)))
        else:
            entries.insert(position, (2, 4, None, pnp.array(np.eye(4), requires_grad=True)))

        assert strategy.build_plan(self._record(entries), N_COLUMNS, "float64") is None

    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_plan_gradcheck(
        self, strategy: BulkContractionStrategy, batch_size: Optional[int], rng: np.random.Generator
    ):
        circuit = [
            (SptmCompRyRy, 0),
            (SptmCompRzRz, 4),
            (SptmCompRxRx, 2),
            (SptmCompRyRy, 2),
            (SptmCompRzRz, 8),
            (SptmCompRxRx, 6),
            (SptmCompRyRy, 0),
        ]
        angles_shape = (len(circuit), 2) if batch_size is None else (len(circuit), batch_size, 2)
        angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=angles_shape)).requires_grad_()
        block = np.linalg.qr(rng.normal(size=(6, 6)))[0]

        def product_of(angles_input: torch.Tensor) -> torch.Tensor:
            records = SptmCircuitRecords()
            for (gate_class, start), gate_angles in zip(circuit, angles_input):
                records.add_gate(gate_class, start, 4, gate_angles)
                if start == 2:
                    records.add_blocks([(4, block)])
                    records.add_gate(SptmCompZX, 0, 4, np.zeros(0))
            groups = strategy.build_plan(records, 12, "float64")
            assert groups is not None
            return SingleParticleTransitionMatrixOperation.multiply_groups(torch.eye(12, dtype=torch.float64), groups)

        assert product_of(angles).requires_grad
        assert torch.autograd.gradcheck(
            product_of, (angles,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    @pytest.mark.parametrize("batching", ["unbatched", "batched", "mixed"])
    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_plan_gradients_are_those_of_the_dense_product(
        self, strategy: BulkContractionStrategy, batching: str, dtype_name: str, rng: np.random.Generator
    ):
        entries = self._random_entries(rng, "torch", batching, n_entries=16)
        weights = torch.from_numpy(rng.normal(size=(N_COLUMNS, N_COLUMNS)))
        plan_leaves = [
            tensor.clone().requires_grad_() for _, _, gate_class, tensor in entries if gate_class in ROTATIONS.values()
        ]
        dense_leaves = [leaf.detach().clone().requires_grad_() for leaf in plan_leaves]
        plan_entries: List[Any] = []
        dense_entries: List[Any] = []
        for leaves, target in ((plan_leaves, plan_entries), (dense_leaves, dense_entries)):
            remaining = iter(leaves)
            for start, size, gate_class, tensor in entries:
                is_rotation = gate_class in ROTATIONS.values()
                target.append((start, size, gate_class, next(remaining) if is_rotation else tensor))

        groups = strategy.build_plan(self._record(plan_entries), N_COLUMNS, dtype_name)
        assert groups is not None
        product = SingleParticleTransitionMatrixOperation.multiply_groups(
            torch.eye(N_COLUMNS, dtype=getattr(torch, dtype_name)), groups
        )
        gradients = torch.autograd.grad((product * weights).sum(), plan_leaves)
        dense = torch.eye(N_COLUMNS, dtype=torch.float64)
        for start, size, gate_class, tensor in dense_entries:
            tensor = torch.as_tensor(tensor, dtype=torch.float64)
            block = tensor if gate_class is None else gate_class.compute_sptm(tensor)
            eye = torch.eye(N_COLUMNS, dtype=torch.float64).expand(*block.shape[:-2], N_COLUMNS, N_COLUMNS)
            stop = start + size
            padded = torch.cat(
                [
                    eye[..., :, :start],
                    torch.cat([eye[..., :start, start:stop], block, eye[..., stop:, start:stop]], dim=-2),
                    eye[..., :, stop:],
                ],
                dim=-1,
            )
            dense = dense @ padded
        expected_gradients = torch.autograd.grad((dense * weights).sum(), dense_leaves)

        assert len(gradients) > 0
        is_double = dtype_name == "float64"
        atol = ATOL_MATRIX_COMPARISON if is_double else ATOL_APPROX_COMPARISON
        rtol = RTOL_MATRIX_COMPARISON if is_double else RTOL_APPROX_COMPARISON
        for gradient, expected in zip(gradients, expected_gradients):
            assert gradient.dtype == torch.float64
            torch.testing.assert_close(gradient, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("as_gate", [True, False])
    def test_a_batch_of_one_is_broadcast_with_a_larger_batch_of_the_same_kind(
        self, strategy: BulkContractionStrategy, backend: str, as_gate: bool, rng: np.random.Generator
    ):
        if as_gate:
            tensors = [rng.uniform(-np.pi, np.pi, size=(1, 2)), rng.uniform(-np.pi, np.pi, size=(BATCH_SIZE, 2))]
        else:
            tensors = [
                np.linalg.qr(rng.normal(size=(1, 4, 4)))[0],
                np.linalg.qr(rng.normal(size=(BATCH_SIZE, 4, 4)))[0],
            ]
        entries = [
            (
                2 * index,
                4,
                SptmCompRyRy if as_gate else None,
                torch.from_numpy(tensor) if backend == "torch" else tensor,
            )
            for index, tensor in enumerate(tensors)
        ]

        groups = strategy.build_plan(self._record(entries), N_COLUMNS, "float64")

        assert groups is not None
        product = SingleParticleTransitionMatrixOperation.multiply_groups(np.eye(N_COLUMNS), groups)
        np.testing.assert_allclose(
            self._to_numpy(product),
            self._dense_product(entries),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_plan_of_no_records_is_empty(self):
        assert BulkContractionStrategy().build_plan(SptmCircuitRecords(), 8, "float64") == []

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_integer_angles_and_blocks_are_computed_as_floats(self, strategy: BulkContractionStrategy, backend: str):
        angles = np.array([1, 2])
        swap = np.asarray(SptmCompZX(wires=[0, 1]).sptm_matrix())
        entries: list
        if backend == "torch":
            entries = [(0, 4, SptmCompRyRy, torch.from_numpy(angles)), (4, 4, None, torch.from_numpy(swap))]
        else:
            entries = [(0, 4, SptmCompRyRy, angles), (4, 4, None, swap)]
        groups = strategy.build_plan(self._record(entries), N_COLUMNS, "float64")
        assert groups is not None
        product = SingleParticleTransitionMatrixOperation.multiply_groups(np.eye(N_COLUMNS), groups)
        expected = self._dense_product([(0, 4, SptmCompRyRy, angles.astype(float)), (4, 4, None, swap.astype(float))])
        np.testing.assert_allclose(
            self._to_numpy(product), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_a_gate_class_with_a_batch_of_one_takes_the_batch_of_another_class(
        self, strategy: BulkContractionStrategy, backend: str
    ):
        generator = np.random.default_rng(TEST_SEED)
        tensors: list = [generator.uniform(0.0, 2.0, size=(1, 2)), generator.uniform(0.0, 2.0, size=(BATCH_SIZE, 2))]
        if backend == "torch":
            tensors = [torch.from_numpy(tensor) for tensor in tensors]
        entries = [(0, 4, SptmCompRyRy, tensors[0]), (2, 4, SptmCompRzRz, tensors[1])]
        groups = strategy.build_plan(self._record(entries), N_COLUMNS, "float64")
        assert groups is not None
        assert all(tuple(stacked.shape)[0] == BATCH_SIZE for _, stacked in groups)
        product = SingleParticleTransitionMatrixOperation.multiply_groups(np.eye(N_COLUMNS), groups)
        np.testing.assert_allclose(
            self._to_numpy(product),
            np.broadcast_to(self._dense_product(entries), (BATCH_SIZE, N_COLUMNS, N_COLUMNS)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
