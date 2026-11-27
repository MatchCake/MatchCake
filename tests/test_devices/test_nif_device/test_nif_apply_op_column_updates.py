import copy
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from pennylane import numpy as pnp

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.contraction_strategies import contraction_strategy_map
from matchcake.operations import SptmCompRyRy, SptmCompRzRz, SptmCompZX
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils import make_transition_matrix_from_action_matrix

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

N_WIRES = 5
BATCH_SIZE = 3
CONTRACTION_STRATEGIES = [None, *sorted(contraction_strategy_map)]
# (gate name, wires) of every gate, in the order they are applied
CIRCUITS = {
    "single_rotation": [("ryry", [1, 2])],
    "single_swap": [("zx", [3, 4])],
    "same_wires": [("ryry", [0, 1]), ("rzrz", [0, 1]), ("zx", [0, 1]), ("ryry", [0, 1])],
    "ladder": [
        ("ryry", [0, 1]),
        ("rzrz", [1, 2]),
        ("zx", [2, 3]),
        ("ryry", [3, 4]),
        ("rzrz", [2, 3]),
        ("zx", [1, 2]),
        ("ryry", [0, 1]),
    ],
    "brick_wall": [
        ("ryry", [0, 1]),
        ("rzrz", [2, 3]),
        ("zx", [1, 2]),
        ("rzrz", [3, 4]),
        ("zx", [0, 1]),
        ("ryry", [2, 3]),
        ("zx", [3, 4]),
        ("ryry", [1, 2]),
    ],
    "rotation_layers": [
        ("ryry", [0, 1]),
        ("rzrz", [3, 4]),
        ("rzrz", [1, 2]),
        ("ryry", [3, 4]),
        ("ryry", [0, 1]),
        ("rzrz", [2, 3]),
    ],
    "swaps_only": [("zx", [0, 1]), ("zx", [2, 3]), ("zx", [1, 2]), ("zx", [3, 4]), ("zx", [2, 3])],
    "disjoint_layers": [
        ("ryry", [0, 1]),
        ("rzrz", [3, 4]),
        ("zx", [0, 1]),
        ("ryry", [2, 3]),
        ("rzrz", [0, 1]),
        ("ryry", [3, 4]),
    ],
}
GRADCHECK_CIRCUITS = ("same_wires", "ladder", "disjoint_layers")
# which gates have batched parameters, None for every gate
BATCHED_GATES_CASES = [
    pytest.param(None, id="all_batched"),
    pytest.param((0,), id="first_batched"),
    pytest.param((1, 4), id="some_batched"),
    pytest.param((-1,), id="last_batched"),
]


class TestNIFApplyOpColumnUpdates:
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
    def _random_params(
        rng: np.random.Generator,
        n_gates: int,
        batched_gates: Optional[Sequence[int]] = (),
    ) -> List[np.ndarray]:
        batched_indices = range(n_gates) if batched_gates is None else [index % n_gates for index in batched_gates]
        return [
            rng.uniform(-np.pi, np.pi, size=(BATCH_SIZE, 2) if index in batched_indices else (2,))
            for index in range(n_gates)
        ]

    @staticmethod
    def _make_operation(gate_name: str, params: Any, wires: List[int]) -> SingleParticleTransitionMatrixOperation:
        if gate_name == "ryry":
            return SptmCompRyRy(params, wires=wires)
        if gate_name == "rzrz":
            return SptmCompRzRz(params, wires=wires)
        return SptmCompZX(wires=wires)

    @classmethod
    def _make_operations(
        cls,
        circuit: List[Tuple[str, List[int]]],
        params: Sequence[Any],
        backends: Sequence[str],
    ) -> List[SingleParticleTransitionMatrixOperation]:
        return [
            cls._make_operation(gate_name, cls._to_backend(gate_params, backend), wires)
            for (gate_name, wires), gate_params, backend in zip(circuit, params, backends)
        ]

    @classmethod
    def _embed(cls, operation: SingleParticleTransitionMatrixOperation) -> np.ndarray:
        block = cls._to_numpy(operation.sptm_matrix())
        start = 2 * min(operation.wires.tolist())
        padded = np.zeros((*block.shape[:-2], 2 * N_WIRES, 2 * N_WIRES))
        padded[...] = np.eye(2 * N_WIRES)
        padded[..., start : start + block.shape[-2], start : start + block.shape[-1]] = block
        return padded

    @classmethod
    def _dense_product(
        cls,
        operations: Sequence[SingleParticleTransitionMatrixOperation],
        initial_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = np.eye(2 * N_WIRES) if initial_matrix is None else initial_matrix
        for operation in operations:
            result = result @ cls._embed(operation)
        return result

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("circuit_name", list(CIRCUITS))
    def test_global_sptm_is_the_product_of_the_padded_matrices(self, contraction_strategy, backend, circuit_name, rng):
        circuit = CIRCUITS[circuit_name]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert tuple(global_matrix.shape) == (2 * N_WIRES, 2 * N_WIRES)
        assert str(global_matrix.dtype).replace("torch.", "") == "float64"
        assert device.global_sptm.wires == device.wires
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("circuit_name", ["ladder", "brick_wall", "disjoint_layers"])
    @pytest.mark.parametrize("batched_gates", BATCHED_GATES_CASES)
    def test_global_sptm_with_batched_parameters(self, contraction_strategy, backend, circuit_name, batched_gates, rng):
        circuit = CIRCUITS[circuit_name]
        params = self._random_params(rng, len(circuit), batched_gates)
        operations = self._make_operations(circuit, params, [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert tuple(global_matrix.shape) == (BATCH_SIZE, 2 * N_WIRES, 2 * N_WIRES)
        assert device.global_sptm.batch_size == BATCH_SIZE
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize(
        "backends",
        [
            pytest.param(("numpy", "torch"), id="numpy_then_torch"),
            pytest.param(("torch", "numpy"), id="torch_then_numpy"),
        ],
    )
    def test_global_sptm_of_numpy_and_torch_operations_is_torch(self, contraction_strategy, backends, rng):
        circuit = CIRCUITS["brick_wall"]
        gate_backends = [backends[index % 2] for index in range(len(circuit))]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), gate_backends)
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(operations)
        global_matrix = device.global_sptm.sptm_matrix()

        assert isinstance(global_matrix, torch.Tensor)
        np.testing.assert_allclose(
            self._to_numpy(global_matrix),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("r_dtype, dtype_name", [(torch.float32, "float32"), (torch.float64, "float64")])
    @pytest.mark.parametrize("circuit_name", ["swaps_only", "brick_wall"])
    def test_global_sptm_has_the_real_dtype_of_the_device(self, backend, r_dtype, dtype_name, circuit_name, rng):
        circuit = CIRCUITS[circuit_name]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES, r_dtype=r_dtype)

        for operation in operations:
            applied = device.apply_op(operation)
            assert str(applied.sptm_matrix().dtype).replace("torch.", "") == dtype_name

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch", "autograd"])
    @pytest.mark.parametrize("circuit_name", ["same_wires", "ladder", "brick_wall", "swaps_only"])
    @pytest.mark.parametrize("batched_gates", [pytest.param((), id="unbatched"), *BATCHED_GATES_CASES])
    def test_matrices_read_from_global_sptm_keep_their_values(self, backend, circuit_name, batched_gates, rng):
        circuit = CIRCUITS[circuit_name]
        params = self._random_params(rng, len(circuit), batched_gates)
        operations = self._make_operations(circuit, params, [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        read_matrices = [device.global_sptm.sptm_matrix()]
        read_operations = [device.global_sptm]

        for operation in operations:
            device.apply([operation])
            read_matrices.append(device.global_sptm.sptm_matrix())
            read_operations.append(device.global_sptm)

        for n_applied, (read_matrix, read_operation) in enumerate(zip(read_matrices, read_operations)):
            expected = self._dense_product(operations[:n_applied])
            if n_applied == 0:
                expected = np.eye(2 * N_WIRES)
            for matrix in (read_matrix, read_operation.sptm_matrix()):
                np.testing.assert_allclose(
                    self._to_numpy(matrix),
                    np.broadcast_to(expected, tuple(matrix.shape)),
                    atol=ATOL_MATRIX_COMPARISON,
                    rtol=RTOL_MATRIX_COMPARISON,
                )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("n_first", [1, 3, 5])
    def test_matrix_read_between_two_applies_keeps_its_values(self, contraction_strategy, backend, n_first, rng):
        circuit = CIRCUITS["brick_wall"]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(operations[:n_first])
        read_matrix = device.global_sptm.sptm_matrix()
        read_values = self._to_numpy(read_matrix).copy()
        device.apply(operations[n_first:])

        np.testing.assert_array_equal(self._to_numpy(read_matrix), read_values)
        np.testing.assert_allclose(
            read_values,
            self._dense_product(operations[:n_first]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_matrix_returned_by_apply_op_keeps_its_values(self, backend, rng):
        circuit = CIRCUITS["ladder"]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        returned_matrix = device.apply_op(operations[0]).sptm_matrix()
        returned_values = self._to_numpy(returned_matrix).copy()
        device.apply_op(operations[1])

        np.testing.assert_array_equal(self._to_numpy(returned_matrix), returned_values)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("circuit_name", ["ladder", "swaps_only"])
    def test_apply_op_returns_the_updated_global_sptm(self, backend, circuit_name, rng):
        circuit = CIRCUITS[circuit_name]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        for n_applied, operation in enumerate(operations, start=1):
            applied = device.apply_op(operation)

            assert type(applied) is SingleParticleTransitionMatrixOperation
            assert applied.wires == device.wires
            np.testing.assert_allclose(
                self._to_numpy(applied.sptm_matrix()),
                self._dense_product(operations[:n_applied]),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("matrix_backend", ["numpy", "torch"])
    @pytest.mark.parametrize("as_operation", [False, True])
    def test_matrix_given_to_the_global_sptm_setter_keeps_its_values(self, backend, matrix_backend, as_operation, rng):
        circuit = CIRCUITS["ladder"]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        initial_values = rng.normal(size=(2 * N_WIRES, 2 * N_WIRES))
        initial_matrix = self._to_backend(initial_values, matrix_backend)
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        if as_operation:
            device.global_sptm = SingleParticleTransitionMatrixOperation(initial_matrix, wires=range(N_WIRES))
        else:
            device.global_sptm = initial_matrix
        device.apply(operations)

        np.testing.assert_array_equal(self._to_numpy(initial_matrix), initial_values)
        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations, initial_values),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_transition_matrix_follows_the_gates_applied_after_it_was_read(self, backend, rng):
        circuit = CIRCUITS["ladder"]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES)

        device.apply(operations[:3])
        first_transition_matrix = device.transition_matrix
        first_values = self._to_numpy(first_transition_matrix).copy()
        device.apply(operations[3:])

        np.testing.assert_array_equal(self._to_numpy(first_transition_matrix), first_values)
        np.testing.assert_allclose(
            first_values,
            make_transition_matrix_from_action_matrix(self._dense_product(operations[:3])),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._to_numpy(device.transition_matrix),
            make_transition_matrix_from_action_matrix(self._dense_product(operations)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("read_before_reset", [False, True])
    def test_reset_starts_again_from_the_identity(self, backend, read_before_reset, rng):
        circuit = CIRCUITS["ladder"]
        operations = self._make_operations(circuit, self._random_params(rng, len(circuit)), [backend] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply(operations[:4])
        read_matrix = device.global_sptm.sptm_matrix() if read_before_reset else None

        device.reset()

        assert device._pending_layers == []
        np.testing.assert_array_equal(self._to_numpy(device.global_sptm.sptm_matrix()), np.eye(2 * N_WIRES))
        device.apply(operations[4:])
        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations[4:]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        if read_matrix is not None:
            np.testing.assert_allclose(
                self._to_numpy(read_matrix),
                self._dense_product(operations[:4]),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize(
        "backends",
        [
            pytest.param(("autograd", "autograd"), id="autograd"),
            pytest.param(("numpy", "autograd"), id="numpy_then_autograd"),
            pytest.param(("autograd", "numpy"), id="autograd_then_numpy"),
        ],
    )
    @pytest.mark.parametrize("circuit_name", ["single_rotation", "ladder", "rotation_layers"])
    @pytest.mark.parametrize("batched_gates", [pytest.param((), id="unbatched"), *BATCHED_GATES_CASES])
    def test_pennylane_numpy_parameters_fall_back_to_the_padded_product(
        self, contraction_strategy, backends, circuit_name, batched_gates, rng
    ):
        circuit = CIRCUITS[circuit_name]
        gate_backends = [backends[index % 2] for index in range(len(circuit))]
        params = self._random_params(rng, len(circuit), batched_gates)
        operations = self._make_operations(circuit, params, gate_backends)
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(operations)

        expected = self._dense_product(operations)
        global_matrix = device.global_sptm.sptm_matrix()
        assert tuple(global_matrix.shape) == expected.shape
        np.testing.assert_allclose(
            self._to_numpy(global_matrix), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize("batched_gates", [pytest.param((), id="unbatched"), pytest.param(None, id="batched")])
    def test_pennylane_numpy_rotation_merged_with_an_integer_swap_on_lower_wires(
        self, contraction_strategy, batched_gates, rng
    ):
        circuit = [("ryry", [0, 1]), ("zx", [1, 2]), ("rzrz", [3, 4]), ("ryry", [0, 1])]
        params = self._random_params(rng, len(circuit), batched_gates)
        operations = self._make_operations(circuit, params, ["autograd"] * len(circuit))
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(operations)

        np.testing.assert_allclose(
            self._to_numpy(device.global_sptm.sptm_matrix()),
            self._dense_product(operations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize("circuit_name", GRADCHECK_CIRCUITS)
    @pytest.mark.parametrize("read_between_gates", [False, True])
    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_global_sptm_gradcheck(self, contraction_strategy, circuit_name, read_between_gates, batch_size, rng):
        circuit = CIRCUITS[circuit_name]
        params_shape = (len(circuit), 2) if batch_size is None else (len(circuit), batch_size, 2)
        params = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=params_shape)).requires_grad_()

        def global_sptm_matrix(params_input: torch.Tensor) -> torch.Tensor:
            device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)
            operations = [
                self._make_operation(gate_name, gate_params, wires)
                for (gate_name, wires), gate_params in zip(circuit, params_input)
            ]
            if not read_between_gates:
                device.apply(operations)
                return device.global_sptm.sptm_matrix()
            read_matrices = []
            for operation in operations:
                device.apply([operation])
                read_matrices.append(device.global_sptm.sptm_matrix())
            return torch.stack([torch.broadcast_to(matrix, read_matrices[-1].shape) for matrix in read_matrices])

        assert torch.autograd.gradcheck(
            global_sptm_matrix,
            (params,),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    @pytest.mark.parametrize("contraction_strategy", CONTRACTION_STRATEGIES)
    @pytest.mark.parametrize("circuit_name", GRADCHECK_CIRCUITS)
    def test_global_sptm_gradient_matches_the_padded_product(self, contraction_strategy, circuit_name, rng):
        circuit = CIRCUITS[circuit_name]
        params_values = rng.uniform(-np.pi, np.pi, size=(len(circuit), 2))
        weights = torch.from_numpy(rng.normal(size=(2 * N_WIRES, 2 * N_WIRES)))
        device_params = torch.from_numpy(params_values.copy()).requires_grad_()
        dense_params = torch.from_numpy(params_values.copy()).requires_grad_()
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=contraction_strategy)

        device.apply(
            [
                self._make_operation(gate_name, gate_params, wires)
                for (gate_name, wires), gate_params in zip(circuit, device_params)
            ]
        )
        (device.global_sptm.sptm_matrix() * weights).sum().backward()
        dense = torch.eye(2 * N_WIRES, dtype=torch.float64)
        for (gate_name, wires), gate_params in zip(circuit, dense_params):
            block = self._make_operation(gate_name, gate_params, wires).sptm_matrix()
            start = 2 * min(wires)
            padded = torch.block_diag(
                torch.eye(start, dtype=torch.float64),
                torch.as_tensor(block, dtype=torch.float64),
                torch.eye(2 * N_WIRES - start - block.shape[-1], dtype=torch.float64),
            )
            dense = dense @ padded
        (dense * weights).sum().backward()

        assert device_params.grad is not None
        assert dense_params.grad is not None
        torch.testing.assert_close(
            device_params.grad, dense_params.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_a_shallow_copy_of_the_device_does_not_write_into_the_original(self):
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply([SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1])])
        duplicate = copy.copy(device)
        before = np.array(device.global_sptm.sptm_matrix())
        duplicate.apply_generator(iter([SptmCompRyRy(np.array([0.5, 0.1]), wires=[1, 2])]))
        np.testing.assert_array_equal(np.asarray(device.global_sptm.sptm_matrix()), before)
        device.apply_generator(iter([SptmCompRzRz(np.array([0.2, 0.7]), wires=[0, 1])]))
        assert not np.allclose(
            np.asarray(device.global_sptm.sptm_matrix()), np.asarray(duplicate.global_sptm.sptm_matrix())
        )

    def test_an_overridden_apply_op_is_called_for_every_operation(self):
        calls = []

        class CountingDevice(NonInteractingFermionicDevice):
            def apply_op(self, op):
                calls.append(op)
                return super().apply_op(op)

        device = CountingDevice(wires=N_WIRES, contraction_strategy=None)
        device.apply(
            [SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]), SptmCompRzRz(np.array([0.5, 0.1]), wires=[1, 2])]
        )
        assert len(calls) == 2

    def test_a_batch_of_one_followed_by_a_larger_batch_is_broadcast(self):
        first = SptmCompRyRy(np.array([[0.3, 0.4]]), wires=[0, 1])
        second = SptmCompRzRz(np.array([[0.5, 0.1], [0.2, 0.7], [0.9, 0.6]]), wires=[1, 2])
        device = NonInteractingFermionicDevice(wires=3, contraction_strategy=None)
        device.apply([first, second])
        wires = range(3)
        expected = first.pad(wires).sptm_matrix() @ second.pad(wires).sptm_matrix()
        np.testing.assert_allclose(
            np.asarray(device.global_sptm.sptm_matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_read_without_gradient_mode_keeps_the_gradient_of_the_operations(self):
        params = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([SptmCompRyRy(params, wires=[0, 1])])
        with torch.no_grad():
            matrix = device.global_sptm.sptm_matrix()
        assert matrix.requires_grad
        assert device.global_sptm.sptm_matrix().requires_grad

    def test_an_operation_with_another_batch_size_is_refused_when_it_is_applied(self):
        device = NonInteractingFermionicDevice(wires=3, contraction_strategy=None)
        device.apply([SptmCompRyRy(np.full((3, 2), 0.3), wires=[0, 1])])
        with pytest.raises(ValueError, match="batch size"):
            device.apply_generator(iter([SptmCompRzRz(np.full((4, 2), 0.5), wires=[1, 2])]))

    def test_an_operation_that_does_not_fit_in_the_matrix_is_refused_when_it_is_applied(self):
        device = NonInteractingFermionicDevice(wires=2)
        with pytest.raises(ValueError, match="columns"):
            device._check_blocks([(2, np.eye(4))], 4)

    def test_a_failed_read_keeps_the_operations_that_were_applied(self, monkeypatch):
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1])])

        def failing_product(matrix, layers):
            raise RuntimeError("no product")

        monkeypatch.setattr(SingleParticleTransitionMatrixOperation, "multiply_layers", staticmethod(failing_product))
        with pytest.raises(RuntimeError, match="no product"):
            device.global_sptm
        monkeypatch.undo()
        expected = SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]).pad(range(3)).sptm_matrix()
        np.testing.assert_allclose(
            np.asarray(device.global_sptm.sptm_matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_a_matrix_given_by_the_user_can_be_written_into_after_it_is_applied(self, backend):
        first = np.array(SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]).sptm_matrix())
        second = np.array(SptmCompRzRz(np.array([0.5, 0.1]), wires=[0, 1]).sptm_matrix())
        expected = first @ second
        matrices = [torch.from_numpy(first), torch.from_numpy(second)] if backend == "torch" else [first, second]
        device = NonInteractingFermionicDevice(wires=2, contraction_strategy=None)
        device.apply([SingleParticleTransitionMatrixOperation(matrix, wires=[0, 1]) for matrix in matrices])
        matrices[0][...] = 0.0
        matrices[1][...] = 0.0
        np.testing.assert_allclose(
            np.asarray(device.global_sptm.sptm_matrix()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_device_saved_before_the_lazy_product_can_be_restored(self):
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1])])
        expected = np.array(device.global_sptm.sptm_matrix())
        state = {key: value for key, value in device.__dict__.items() if key not in ("_pending_layers", "_batch_size")}
        restored = NonInteractingFermionicDevice.__new__(NonInteractingFermionicDevice)
        restored.__setstate__(state)
        assert restored._pending_layers == [] and restored._batch_size is None
        np.testing.assert_array_equal(np.asarray(restored.global_sptm.sptm_matrix()), expected)

    def test_vectorized_jacobian_through_the_device(self):
        def matrix_of(angles: torch.Tensor) -> torch.Tensor:
            device = NonInteractingFermionicDevice(wires=3)
            device.apply([SptmCompRyRy(angles[:2], wires=[0, 1]), SptmCompRzRz(angles[2:], wires=[1, 2])])
            return device.global_sptm.sptm_matrix()

        angles = torch.tensor([0.3, 0.4, 0.5, 0.1], dtype=torch.float64)
        torch.testing.assert_close(
            torch.autograd.functional.jacobian(matrix_of, angles, vectorize=True),
            torch.autograd.functional.jacobian(matrix_of, angles),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
