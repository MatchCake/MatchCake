from typing import Any

import numpy as np
import pytest
import torch

from matchcake.devices.sptm_circuit_records import SptmCircuitRecords
from matchcake.operations.single_particle_transition_matrices import (
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
)

from ..configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED

BATCH_SIZE = 3


class TestSptmCircuitRecords:
    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    def test_new_records_are_empty(self):
        records = SptmCircuitRecords()

        assert len(records) == 0
        assert records.entries == []
        assert records.block_layers("float64") == []

    def test_add_gate_records_the_class_and_the_parameters(self, rng: np.random.Generator):
        records = SptmCircuitRecords()
        angles = rng.uniform(-np.pi, np.pi, size=2)
        batched_angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=(BATCH_SIZE, 2)))

        records.add_gate(SptmCompRyRy, 2, 4, angles)
        records.add_gate(SptmCompRzRz, 0, 4, batched_angles)

        assert len(records) == 2
        assert records.entries[0][:3] == (2, 4, SptmCompRyRy)
        assert records.entries[0][3] is angles
        assert records.entries[1][:3] == (0, 4, SptmCompRzRz)
        assert records.entries[1][3] is batched_angles

    def test_add_blocks_records_every_block_with_its_size(self, rng: np.random.Generator):
        records = SptmCircuitRecords()
        small, large = rng.normal(size=(4, 4)), torch.from_numpy(rng.normal(size=(BATCH_SIZE, 6, 6)))

        records.add_blocks([(0, small), (6, large)])

        assert len(records) == 2
        assert records.entries[0][:3] == (0, 4, None)
        assert records.entries[0][3] is small
        assert records.entries[1][:3] == (6, 6, None)
        assert records.entries[1][3] is large

    def test_entries_are_kept_in_the_order_they_are_added(self, rng: np.random.Generator):
        records = SptmCircuitRecords()

        records.add_gate(SptmCompRxRx, 4, 4, rng.uniform(size=2))
        records.add_blocks([(0, np.eye(4)), (8, np.eye(6))])
        records.add_gate(SptmCompZX, 2, 4, np.zeros(0))
        records.add_blocks([(4, np.eye(4))])

        assert [(start, size, gate_class) for start, size, gate_class, _ in records.entries] == [
            (4, 4, SptmCompRxRx),
            (0, 4, None),
            (8, 6, None),
            (2, 4, SptmCompZX),
            (4, 4, None),
        ]
        assert len(records) == 5

    def test_clear_forgets_everything(self, rng: np.random.Generator):
        records = SptmCircuitRecords()
        records.add_gate(SptmCompRyRy, 0, 4, rng.uniform(size=2))
        records.add_blocks([(0, np.eye(4)), (4, np.eye(4))])

        records.clear()

        assert len(records) == 0
        assert records.entries == []
        assert records.block_layers("float64") == []
        records.add_blocks([(2, np.eye(4))])
        assert [[start for start, _ in layer] for layer in records.block_layers("float64")] == [[2]]

    def test_block_layers_keeps_the_blocks_of_one_operation_in_one_layer(self, rng: np.random.Generator):
        records = SptmCircuitRecords()
        angles = rng.uniform(-np.pi, np.pi, size=(3, 2))
        blocks = [rng.normal(size=(4, 4)), rng.normal(size=(6, 6)), rng.normal(size=(4, 4))]

        records.add_gate(SptmCompRyRy, 0, 4, angles[0])
        records.add_blocks([(0, blocks[0]), (6, blocks[1])])
        records.add_gate(SptmCompRzRz, 2, 4, angles[1])
        records.add_gate(SptmCompZX, 4, 4, np.zeros(0))
        records.add_blocks([(8, blocks[2])])
        records.add_blocks([])
        records.add_gate(SptmCompRxRx, 6, 4, angles[2])
        layers = records.block_layers("float64")

        expected = [
            [(0, SptmCompRyRy.compute_sptm(angles[0]))],
            [(0, blocks[0]), (6, blocks[1])],
            [(2, SptmCompRzRz.compute_sptm(angles[1]))],
            [(4, SptmCompZX.compute_sptm(np.zeros(0)))],
            [(8, blocks[2])],
            [],
            [(6, SptmCompRxRx.compute_sptm(angles[2]))],
        ]
        assert [[start for start, _ in layer] for layer in layers] == [
            [start for start, _ in layer] for layer in expected
        ]
        for layer, expected_layer in zip(layers, expected):
            for (_, block), (_, expected_block) in zip(layer, expected_layer):
                np.testing.assert_allclose(
                    block, expected_block, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
                )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("stored_dtype", ["float32", "float64", "int64"])
    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_block_layers_casts_the_blocks(
        self, backend: str, stored_dtype: str, dtype_name: str, rng: np.random.Generator
    ):
        records = SptmCircuitRecords()
        angles = rng.uniform(-np.pi, np.pi, size=(BATCH_SIZE, 2)).astype(
            "float32" if stored_dtype == "int64" else stored_dtype
        )
        block = rng.integers(-3, 3, size=(6, 6)).astype(stored_dtype)
        if backend == "torch":
            records.add_gate(SptmCompRyRy, 0, 4, torch.from_numpy(angles))
            records.add_blocks([(2, torch.from_numpy(block))])
        else:
            records.add_gate(SptmCompRyRy, 0, 4, angles)
            records.add_blocks([(2, block)])
        records.add_gate(SptmCompZX, 4, 4, np.zeros(0))

        layers = records.block_layers(dtype_name)

        assert [len(layer) for layer in layers] == [1, 1, 1]
        for (_, layer_block), expected_shape in zip(
            [layer[0] for layer in layers], [(BATCH_SIZE, 4, 4), (6, 6), (4, 4)]
        ):
            assert str(layer_block.dtype).replace("torch.", "") == dtype_name
            assert tuple(layer_block.shape) == expected_shape
        assert isinstance(layers[0][0][1], torch.Tensor if backend == "torch" else np.ndarray)
        np.testing.assert_allclose(
            self._to_numpy(layers[0][0][1]),
            SptmCompRyRy.compute_sptm(angles.astype(np.float64)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_array_equal(self._to_numpy(layers[1][0][1]), block)

    def test_block_layers_keeps_the_gradient_of_the_angles(self, rng: np.random.Generator):
        records = SptmCircuitRecords()
        angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=2)).requires_grad_()
        records.add_gate(SptmCompRzRz, 0, 4, angles)

        ((_, block),) = records.block_layers("float32")[0]
        (gradient,) = torch.autograd.grad(block.sum(), angles)

        assert block.dtype == torch.float32
        assert bool(torch.any(gradient != 0))
