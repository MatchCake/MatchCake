from typing import Any, Iterator, List, Optional, Sequence

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane import numpy as pnp

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.contraction_strategies import BulkContractionStrategy, contraction_strategy_map
from matchcake.devices.sptm_circuit_records import SptmCircuitRecords
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
)

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

N_WIRES = 6
N_GATES = 24
BATCH_SIZE = 3
STRATEGIES = sorted(contraction_strategy_map)
ROTATIONS = {"rxrx": SptmCompRxRx, "ryry": SptmCompRyRy, "rzrz": SptmCompRzRz}
MATCHGATES = {"rxrx": CompRxRx, "ryry": CompRyRy, "rzrz": CompRzRz}
# gate kind -> number of wires it spans
GATE_KINDS = {
    "rxrx": 2,
    "ryry": 2,
    "rzrz": 2,
    "reversed_rzrz": 2,
    "zx": 2,
    "distant_zx": 3,
    "plain_two_wires": 2,
    "plain_three_wires": 3,
    "matchgate": 2,
}


class TestNIFBulkRecords:
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

    @classmethod
    def _random_operations(
        cls,
        rng: np.random.Generator,
        backend: str,
        batching: str = "unbatched",
        wire_offset: int = 0,
        n_gates: int = N_GATES,
        kinds: Optional[Sequence[str]] = None,
    ) -> List[qml.operation.Operation]:
        kinds = list(GATE_KINDS) if kinds is None else list(kinds)
        operations: List[qml.operation.Operation] = []
        for _ in range(n_gates):
            kind = kinds[int(rng.integers(len(kinds)))]
            span = GATE_KINDS[kind]
            first_wire = wire_offset + int(rng.integers(N_WIRES - span + 1))
            wires = list(range(first_wire, first_wire + span))
            is_batched = batching == "batched" or (batching == "mixed" and bool(rng.integers(2)))
            batch_shape = (BATCH_SIZE,) if is_batched else ()
            angles = cls._to_backend(rng.uniform(-np.pi, np.pi, size=(*batch_shape, 2)), backend)
            if kind in ROTATIONS:
                operations.append(ROTATIONS[kind](angles, wires=wires))
            elif kind == "reversed_rzrz":
                operations.append(SptmCompRzRz(angles, wires=wires[::-1]))
            elif kind == "zx":
                operations.append(SptmCompZX(wires=wires))
            elif kind == "distant_zx":
                operations.append(SptmCompZX(wires=[wires[0], wires[-1]]))
            elif kind == "matchgate":
                operations.append(CompRyRy(angles, wires=wires))
            else:
                orthogonal, _ = np.linalg.qr(rng.normal(size=(*batch_shape, 2 * span, 2 * span)))
                operations.append(
                    SingleParticleTransitionMatrixOperation(cls._to_backend(orthogonal, backend), wires=wires)
                )
        return operations

    @classmethod
    def _dense_product(
        cls,
        operations: Sequence[qml.operation.Operation],
        wire_offset: int = 0,
        initial_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = np.eye(2 * N_WIRES) if initial_matrix is None else initial_matrix
        for operation in operations:
            sptm_operation = SingleParticleTransitionMatrixOperation.from_operation(operation)
            block = np.real(cls._to_numpy(sptm_operation.sptm_matrix())).astype(np.float64)
            start = 2 * (min(operation.wires.tolist()) - wire_offset)
            padded = np.zeros((*block.shape[:-2], 2 * N_WIRES, 2 * N_WIRES))
            padded[...] = np.eye(2 * N_WIRES)
            padded[..., start : start + block.shape[-1], start : start + block.shape[-1]] = block
            result = result @ padded
        return result

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    def test_default_strategy_is_bulk(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        assert NonInteractingFermionicDevice.DEFAULT_CONTRACTION_METHOD == "bulk"
        assert isinstance(device.contraction_strategy, BulkContractionStrategy)
        assert isinstance(device._records, SptmCircuitRecords)
        assert len(device._records) == 0

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("r_dtype, dtype_name", [(torch.float32, "float32"), (torch.float64, "float64")])
    @pytest.mark.parametrize("wire_offset", [0, 3])
    def test_global_sptm_is_the_product_of_the_padded_matrices(
        self, strategy_name: str, backend: str, r_dtype, dtype_name: str, wire_offset: int, rng: np.random.Generator
    ):
        operations = self._random_operations(rng, backend, wire_offset=wire_offset)
        device = NonInteractingFermionicDevice(
            wires=range(wire_offset, wire_offset + N_WIRES), contraction_strategy=strategy_name, r_dtype=r_dtype
        )

        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert isinstance(global_matrix, torch.Tensor if backend == "torch" else np.ndarray)
        assert tuple(global_matrix.shape) == (2 * N_WIRES, 2 * N_WIRES)
        assert str(global_matrix.dtype).replace("torch.", "") == dtype_name
        assert device.global_sptm.wires == device.wires
        assert len(device._records) == 0
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations, wire_offset),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("batching", ["batched", "mixed"])
    @pytest.mark.parametrize("r_dtype", [torch.float32, torch.float64])
    def test_global_sptm_with_batched_parameters(
        self, strategy_name: str, backend: str, batching: str, r_dtype, rng: np.random.Generator
    ):
        operations = self._random_operations(rng, backend, batching=batching)
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name, r_dtype=r_dtype)

        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert tuple(global_matrix.shape) == (BATCH_SIZE, 2 * N_WIRES, 2 * N_WIRES)
        assert global_matrix.dtype == (r_dtype if backend == "torch" else np.dtype(str(r_dtype).replace("torch.", "")))
        assert device.global_sptm.batch_size == BATCH_SIZE
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    def test_global_sptm_of_numpy_and_torch_operations_is_torch(self, strategy_name: str, rng: np.random.Generator):
        operations = [
            operation
            for pair in zip(
                self._random_operations(rng, "numpy", n_gates=N_GATES // 2),
                self._random_operations(rng, "torch", n_gates=N_GATES // 2),
            )
            for operation in pair
        ]
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name)

        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert isinstance(global_matrix, torch.Tensor)
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("batching", ["unbatched", "mixed"])
    def test_operations_applied_from_a_generator(
        self, strategy_name: str, backend: str, batching: str, rng: np.random.Generator
    ):
        operations = self._random_operations(rng, backend, batching=batching, wire_offset=2)
        yielded = []

        def generate_operations() -> Iterator[qml.operation.Operation]:
            for operation in operations:
                yielded.append(operation)
                yield operation

        device = NonInteractingFermionicDevice(wires=range(2, 2 + N_WIRES), contraction_strategy=strategy_name)

        returned = device.apply_generator(generate_operations())

        assert returned is device
        assert len(yielded) == len(operations)
        assert device.apply_metadata["n_operations"] == len(operations)
        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            np.broadcast_to(self._dense_product(operations, 2), tuple(device.global_sptm.sptm_matrix().shape)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("chunk_sizes", [(1, 1, 22), (8, 8, 8), (5, 0, 19), (23, 1)])
    def test_several_applies_with_reads_in_between(
        self, strategy_name: str, backend: str, chunk_sizes: Sequence[int], rng: np.random.Generator
    ):
        operations = self._random_operations(rng, backend)
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name)
        read_matrices = []
        n_applied = 0

        for index, chunk_size in enumerate(chunk_sizes):
            chunk = operations[n_applied : n_applied + chunk_size]
            n_applied += chunk_size
            if index % 2 == 0:
                device.apply(chunk)
            else:
                device.apply_generator(operation for operation in chunk)
            read_matrices.append((n_applied, device.global_sptm.sptm_matrix()))
            assert len(device._records) == 0

        assert n_applied == len(operations)
        for n_read, matrix in read_matrices:
            np.testing.assert_allclose(
                self._to_numpy(matrix),
                self._dense_product(operations[:n_read]),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_gates_with_a_formula_are_recorded_by_class_and_parameters(self, backend: str, rng: np.random.Generator):
        angles = [self._to_backend(rng.uniform(-np.pi, np.pi, size=2), backend) for _ in range(3)]
        batched_angles = self._to_backend(rng.uniform(-np.pi, np.pi, size=(BATCH_SIZE, 2)), backend)
        operations = [
            SptmCompRxRx(angles[0], wires=[3, 4]),
            SptmCompRyRy(angles[1], wires=[5, 6]),
            SptmCompRzRz(angles[2], wires=[7, 6]),
            SptmCompZX(wires=[4, 5]),
            SptmCompRyRy(batched_angles, wires=[6, 7]),
        ]
        device = NonInteractingFermionicDevice(wires=range(3, 8))

        device.apply(operations)

        assert [(start, size, gate_class) for start, size, gate_class, _ in device._records.entries] == [
            (0, 4, SptmCompRxRx),
            (4, 4, SptmCompRyRy),
            (6, 4, SptmCompRzRz),
            (2, 4, SptmCompZX),
            (6, 4, SptmCompRyRy),
        ]
        for (_, _, _, params), operation in zip(device._records.entries, operations):
            # the angles are recorded as a copy, so that writing into them later does not change the product
            assert params is not operation._params
            np.testing.assert_array_equal(np.asarray(params), np.asarray(operation._params))
        assert device._batch_size == BATCH_SIZE
        assert all(
            operation._stored_matrix is None for operation in operations if not isinstance(operation, SptmCompZX)
        )
        device.global_sptm
        assert len(device._records) == 0
        assert all(
            operation._stored_matrix is None for operation in operations if not isinstance(operation, SptmCompZX)
        )

    def test_other_operations_are_recorded_by_their_blocks(self, rng: np.random.Generator):
        orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        operations = [
            SptmCompZX(wires=[0, 2]),
            SingleParticleTransitionMatrixOperation(orthogonal, wires=[1, 2, 3]),
            CompRyRy(rng.uniform(size=2), wires=[3, 4]),
            SptmCompRyRy(rng.uniform(size=2), wires=[0, 1]),
        ]
        device = NonInteractingFermionicDevice(wires=N_WIRES, r_dtype=torch.float32)

        device.apply(operations)

        entries = device._records.entries
        assert [(start, size, gate_class) for start, size, gate_class, _ in entries] == [
            (0, 6, None),
            (2, 6, None),
            (6, 4, SptmCompRyRy),
            (0, 4, SptmCompRyRy),
        ]
        assert all(str(block.dtype) == "float32" for _, _, gate_class, block in entries if gate_class is None)
        np.testing.assert_allclose(entries[1][3], orthogonal, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        assert not np.shares_memory(entries[1][3], orthogonal)

    def test_a_pennylane_numpy_gate_is_multiplied_right_away(self, rng: np.random.Generator):
        operations = [
            SptmCompRyRy(rng.uniform(size=2), wires=[0, 1]),
            SptmCompRzRz(pnp.array(rng.uniform(size=2), requires_grad=True), wires=[1, 2]),
            SptmCompRxRx(rng.uniform(size=2), wires=[2, 3]),
        ]
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        device.apply(operations[:2])
        assert len(device._records) == 0
        device.apply(operations[2:])
        assert len(device._records) == 1

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("r_dtype, dtype_name", [(torch.float32, "float32"), (torch.float64, "float64")])
    @pytest.mark.parametrize("batching", ["unbatched", "mixed"])
    def test_a_stored_pennylane_numpy_matrix_falls_back_to_the_padded_product(
        self, strategy_name: str, r_dtype, dtype_name: str, batching: str, rng: np.random.Generator
    ):
        operations = self._random_operations(rng, "numpy", batching=batching, n_gates=12)
        initial_values, _ = np.linalg.qr(rng.normal(size=(2 * N_WIRES, 2 * N_WIRES)))
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name, r_dtype=r_dtype)

        device.global_sptm = pnp.array(initial_values, requires_grad=True)
        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert qml.math.get_interface(global_matrix) == "autograd"
        assert str(global_matrix.dtype) == dtype_name
        assert len(device._records) == 0
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations, initial_matrix=initial_values),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_transition_matrix_follows_the_gates_recorded_after_it_was_read(
        self, backend: str, rng: np.random.Generator
    ):
        operations = self._random_operations(rng, backend, kinds=["rxrx", "ryry", "rzrz", "zx"], n_gates=8)
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        device.apply(operations[:4])
        first_values = self._to_numpy(device.transition_matrix).copy()
        assert device._transition_matrix is not None
        device.apply(operations[4:])

        assert device._transition_matrix is None
        assert len(device._records) == 4
        second_values = self._to_numpy(device.transition_matrix)
        reference = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy="none")
        reference.apply(operations)
        np.testing.assert_allclose(
            second_values,
            self._to_numpy(reference.transition_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        assert not np.allclose(first_values, second_values, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize(
        "start, block_size",
        [(-2, 4), (2 * N_WIRES - 2, 4), (2 * N_WIRES, 4), (0, 2 * N_WIRES + 2), (8, 6)],
    )
    def test_a_block_out_of_the_matrix_is_refused(self, start: int, block_size: int):
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        with pytest.raises(ValueError, match=f"columns {start} to {start + block_size} .* of size {2 * N_WIRES}"):
            device._check_block(start, block_size, None, 2 * N_WIRES)
        assert device._batch_size is None

    @pytest.mark.parametrize("start, block_size", [(0, 4), (2 * N_WIRES - 4, 4), (0, 2 * N_WIRES), (6, 6)])
    @pytest.mark.parametrize("batch_size, expected_batch_size", [(None, None), (1, None), (BATCH_SIZE, BATCH_SIZE)])
    def test_a_block_in_the_matrix_is_accepted(
        self, start: int, block_size: int, batch_size: Optional[int], expected_batch_size: Optional[int]
    ):
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        device._check_block(start, block_size, batch_size, 2 * N_WIRES)

        assert device._batch_size == expected_batch_size

    def test_a_block_with_another_batch_size_is_refused(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        device._check_block(0, 4, BATCH_SIZE, 2 * N_WIRES)

        with pytest.raises(ValueError, match=f"batch size of {BATCH_SIZE + 1}.*batch size of {BATCH_SIZE}"):
            device._check_block(0, 4, BATCH_SIZE + 1, 2 * N_WIRES)
        device._check_block(0, 4, 1, 2 * N_WIRES)
        device._check_block(0, 4, None, 2 * N_WIRES)
        device._check_block(0, 4, BATCH_SIZE, 2 * N_WIRES)

        assert device._batch_size == BATCH_SIZE

    @pytest.mark.parametrize("strategy_name", ["bulk", "none"])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("as_gate", [True, False])
    def test_an_operation_with_another_batch_size_is_refused_when_it_is_applied(
        self, strategy_name: str, backend: str, as_gate: bool, rng: np.random.Generator
    ):
        first = SptmCompRyRy(self._to_backend(rng.uniform(size=(BATCH_SIZE, 2)), backend), wires=[0, 1])
        second: qml.operation.Operation
        if as_gate:
            second = SptmCompRzRz(self._to_backend(rng.uniform(size=(BATCH_SIZE + 1, 2)), backend), wires=[1, 2])
        else:
            second = SingleParticleTransitionMatrixOperation(
                self._to_backend(np.broadcast_to(np.eye(6), (BATCH_SIZE + 1, 6, 6)), backend), wires=[1, 2, 3]
            )
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name)
        device.apply([first])

        with pytest.raises(ValueError, match="batch size"):
            device.apply([second])

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product([first]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_gate_out_of_the_wires_of_the_device_is_refused_when_it_is_applied(self):
        device = NonInteractingFermionicDevice(wires=[0, 1, 2])

        with pytest.raises(ValueError, match="columns"):
            device._apply_op(SptmCompRyRy(np.array([0.3, 0.4]), wires=[2, 3]))
        assert len(device._records) == 0

    @pytest.mark.parametrize(
        "wires, expected_wires",
        [(4, [0, 1, 2, 3]), ([2, 3, 4], [2, 3, 4]), ([1, 4], [1, 2, 3, 4]), ([5, 3, 4], [3, 4, 5])],
    )
    def test_continuous_wires_are_computed_once(self, wires, expected_wires: List[int]):
        device = NonInteractingFermionicDevice(wires=wires)

        continuous_wires = device._continuous_wires

        assert continuous_wires.tolist() == expected_wires
        assert device._continuous_wires is continuous_wires
        assert device._wire_positions == {wire: position for position, wire in enumerate(expected_wires)}
        assert device._wire_positions is device._wire_positions
        device._continuous_wires_cache = None
        assert device._wire_positions == {wire: position for position, wire in enumerate(expected_wires)}
        assert device._continuous_wires == continuous_wires

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    def test_a_device_with_a_gap_in_its_wires_applies_gates_on_their_continuous_span(
        self, strategy_name: str, rng: np.random.Generator
    ):
        operations = [
            SptmCompRyRy(rng.uniform(size=2), wires=[1, 2]),
            SptmCompZX(wires=[1, 4]),
            SptmCompRzRz(rng.uniform(size=2), wires=[3, 4]),
        ]
        device = NonInteractingFermionicDevice(wires=[1, 2, 4], contraction_strategy=strategy_name)
        expected = np.eye(8)
        for operation in operations:
            expected = expected @ np.asarray(operation.sptm_matrix(range(1, 5)))

        device.apply(operations)

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_a_rotation_on_distant_wires_is_recorded_by_its_block(
        self, strategy_name: str, backend: str, rng: np.random.Generator
    ):
        angles = rng.uniform(-np.pi, np.pi, size=2)
        operation = SptmCompRyRy(self._to_backend(angles, backend), wires=[1, 3])
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name)
        expected = np.eye(2 * N_WIRES)
        expected[2:6, 2:6] = SptmCompRyRy.compute_sptm(angles)

        device.apply([operation])

        if strategy_name == "bulk":
            assert [(start, size, gate_class) for start, size, gate_class, _ in device._records.entries] == [
                (2, 4, None)
            ]
        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_device_saved_with_pending_layers_is_restored_with_empty_records(self, rng: np.random.Generator):
        operations = self._random_operations(rng, "numpy", n_gates=6)
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply(operations[:3])
        device.global_sptm
        state = {
            key: value
            for key, value in device.__dict__.items()
            if key not in ("_records", "_continuous_wires_cache", "_batch_size")
        }
        state["_pending_layers"] = []
        restored = NonInteractingFermionicDevice.__new__(NonInteractingFermionicDevice)

        restored.__setstate__(state)

        assert "_pending_layers" not in restored.__dict__
        assert isinstance(restored._records, SptmCircuitRecords)
        assert len(restored._records) == 0
        assert restored._records is not device._records
        assert restored._continuous_wires_cache is None
        assert restored._batch_size is None
        restored.apply(operations[3:])
        np.testing.assert_allclose(
            self._to_numpy(restored.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_device_saved_with_records_keeps_them(self, rng: np.random.Generator):
        operations = self._random_operations(rng, "numpy", n_gates=6)
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply(operations)
        restored = NonInteractingFermionicDevice.__new__(NonInteractingFermionicDevice)

        restored.__setstate__(dict(device.__dict__))

        assert restored._records is device._records
        assert len(restored._records) > 0
        np.testing.assert_allclose(
            self._to_numpy(restored.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("gate_family", ["matchgate", "sptm"])
    @pytest.mark.parametrize("measurement", ["expval", "probs"])
    def test_qnode_results_are_those_of_default_qubit(
        self, strategy_name: str, gate_family: str, measurement: str, rng: np.random.Generator
    ):
        n_wires = 4
        layout = [("rxrx", [0, 1]), ("ryry", [2, 3]), ("zx", [1, 2]), ("rzrz", [1, 2]), ("ryry", [0, 1])]
        angles = rng.uniform(0.1, 3.0, size=(len(layout), 2))

        def circuit():
            qml.BasisState(np.array([1, 0, 1, 0]), wires=range(n_wires))
            for (gate_name, wires), gate_angles in zip(layout, angles):
                if gate_name == "zx":
                    (fSWAP if gate_family == "matchgate" else SptmCompZX)(wires=wires)
                elif gate_family == "matchgate":
                    MATCHGATES[gate_name](gate_angles, wires=wires)
                else:
                    ROTATIONS[gate_name](gate_angles, wires=wires)
            if measurement == "expval":
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            return qml.probs(wires=range(n_wires))

        nif_device = NonInteractingFermionicDevice(wires=n_wires, contraction_strategy=strategy_name)
        result = qml.QNode(circuit, nif_device)()
        expected = qml.QNode(circuit, qml.device("default.qubit", wires=n_wires))()

        np.testing.assert_allclose(
            self._to_numpy(result),
            self._to_numpy(expected),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_a_batch_of_one_is_broadcast_with_a_larger_batch_of_the_same_gate_class(
        self, backend: str, rng: np.random.Generator
    ):
        operations = [
            SptmCompRyRy(self._to_backend(rng.uniform(size=(1, 2)), backend), wires=[0, 1]),
            SptmCompRyRy(self._to_backend(rng.uniform(size=(BATCH_SIZE, 2)), backend), wires=[1, 2]),
        ]
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy="bulk")

        device.apply(operations)

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_angles_given_by_the_user_can_be_written_into_after_the_gate_is_applied(self, backend: str):
        angles = self._to_backend(np.array([0.3, 0.4]), backend)
        expected = SptmCompRyRy.compute_sptm(np.array([0.3, 0.4]))
        device = NonInteractingFermionicDevice(wires=2)

        device.apply([SptmCompRyRy(angles, wires=[0, 1])])
        angles[...] = 0.0

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("scale", [1.0, 1e2, 1e3])
    def test_large_angles_keep_single_precision_accuracy_on_a_single_precision_device(self, scale: float):
        """The angles are double precision: casting them before the cosine would lose their low bits."""
        angles = scale * np.array([[0.3, 0.4], [0.7, 0.2]])
        gates = [SptmCompRyRy(angles[0], wires=[0, 1]), SptmCompRzRz(angles[1], wires=[1, 2])]
        matrices = []
        for r_dtype in (torch.float32, torch.float64):
            device = NonInteractingFermionicDevice(wires=3, r_dtype=r_dtype)
            device.apply(gates)
            matrices.append(np.asarray(device.global_sptm.sptm_matrix()))
        assert matrices[0].dtype == np.float32
        np.testing.assert_allclose(matrices[0], matrices[1], atol=1e-6, rtol=0.0)

    def test_a_batched_gate_does_not_make_the_other_stacks_batched(self):
        """Every stack as large as the batch would make the backward pass grow with the batch and the depth."""
        device = NonInteractingFermionicDevice(wires=4)
        device.apply(
            [
                SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]),
                SptmCompRzRz(np.full((5, 2), 0.2), wires=[2, 3]),
                SptmCompRyRy(np.array([0.1, 0.6]), wires=[1, 2]),
            ]
        )
        groups = BulkContractionStrategy().build_plan(device._records, 8, "float64")
        assert groups is not None
        assert sorted(stacked.ndim for _, stacked in groups) == [3, 3, 4]
        assert tuple(np.asarray(device.global_sptm.sptm_matrix()).shape) == (5, 8, 8)

    def test_a_gate_with_complex_angles_is_applied_as_its_checked_matrix(self):
        angles = np.array([0.3 + 0.0j, 0.5 + 0.0j])
        device = NonInteractingFermionicDevice(wires=2)
        device.apply([SptmCompRxRx(angles, wires=[0, 1])])
        entry = device._records.entries[0]
        assert entry[2] is None
        np.testing.assert_allclose(
            np.asarray(device.global_sptm.sptm_matrix()),
            np.asarray(SptmCompRxRx(np.real(angles), wires=[0, 1]).sptm_matrix()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("gate", ["rotation", "swap"])
    def test_a_gate_on_wires_that_are_not_on_the_device_raises_a_wire_error(self, gate: str):
        device = NonInteractingFermionicDevice(wires=3)
        operation = SptmCompRxRx(np.array([0.3, 0.4]), wires=[5, 6]) if gate == "rotation" else SptmCompZX(wires=[5, 6])
        with pytest.raises(qml.wires.WireError):
            device.apply_generator(iter([operation]))

    def test_operations_pending_in_a_device_saved_before_the_records_are_restored(self):
        gate = SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1])
        device = NonInteractingFermionicDevice(wires=2)
        state = {key: value for key, value in device.__dict__.items() if key != "_records"}
        state["_pending_layers"] = [[(0, np.asarray(gate.sptm_matrix()))]]
        restored = NonInteractingFermionicDevice.__new__(NonInteractingFermionicDevice)
        restored.__setstate__(state)
        assert len(restored._records) == 1
        np.testing.assert_allclose(
            np.asarray(restored.global_sptm.sptm_matrix()),
            np.asarray(gate.sptm_matrix()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_the_global_sptm_stays_on_the_accelerator_when_later_gates_are_on_the_cpu(self):
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([SptmCompRyRy(torch.tensor([0.3, 0.4], dtype=torch.float64, device="cuda"), wires=[0, 1])])
        assert device.global_sptm.sptm_matrix().device.type == "cuda"
        device.apply_generator(iter([SptmCompRzRz(torch.tensor([0.5, 0.1], dtype=torch.float64), wires=[1, 2])]))
        assert device.global_sptm.sptm_matrix().device.type == "cuda"
