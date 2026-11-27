from typing import Any, List

import numpy as np
import pytest
import torch
from pennylane.wires import Wires

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.contraction_strategies import (
    NeighboursContractionStrategy,
    contraction_strategy_map,
    get_contraction_strategy,
)
from matchcake.devices.contraction_strategies.forward_strategy import _ForwardMatchgatesContainer
from matchcake.devices.sptm_circuit_records import SptmCircuitRecords
from matchcake.operations import CompRyRy
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
)

from ...configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED

N_WIRES = 4
STRATEGIES = sorted(contraction_strategy_map)
CONTRACTING_STRATEGIES = ["forward", "horizontal", "neighbours", "vertical"]
# (gate name, wires) of every gate, in the order they are applied
CIRCUIT = [
    ("ryry", [0, 1]),
    ("rzrz", [2, 3]),
    ("rxrx", [1, 2]),
    ("zx", [0, 1]),
    ("ryry", [0, 1]),
    ("rzrz", [2, 3]),
    ("rxrx", [2, 3]),
    ("ryry", [1, 2]),
]
GATE_CLASSES = {"rxrx": SptmCompRxRx, "ryry": SptmCompRyRy, "rzrz": SptmCompRzRz}
HANDED_OPERATIONS_CASES = [
    pytest.param(
        strategy_name,
        backend,
        id=f"{strategy_name}-{backend}",
    )
    for strategy_name in CONTRACTING_STRATEGIES
    for backend in ("numpy", "torch")
]


class TestContractionStrategyDtype:
    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _make_operations(rng: np.random.Generator, backend: str) -> List[SingleParticleTransitionMatrixOperation]:
        operations: List[SingleParticleTransitionMatrixOperation] = []
        for gate_name, wires in CIRCUIT:
            if gate_name == "zx":
                operations.append(SptmCompZX(wires=wires))
                continue
            angles = rng.uniform(-np.pi, np.pi, size=2)
            operations.append(
                GATE_CLASSES[gate_name](torch.from_numpy(angles) if backend == "torch" else angles, wires=wires)
            )
        return operations

    @staticmethod
    def _block_dtype_names(operation: SingleParticleTransitionMatrixOperation) -> List[str]:
        return [str(block.dtype).replace("torch.", "") for _, block in operation.column_blocks(Wires(range(N_WIRES)))]

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    def test_dtype_name_is_none_by_default(self, strategy_name: str):
        strategy = get_contraction_strategy(strategy_name)

        assert strategy.dtype_name is None
        if strategy.container is not None:
            assert strategy.container.dtype_name is None

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("dtype_name", ["float32", "float64", None])
    def test_dtype_name_is_given_to_the_container(self, strategy_name: str, dtype_name):
        strategy = get_contraction_strategy(strategy_name)

        strategy.dtype_name = "float16"
        strategy.dtype_name = dtype_name

        assert strategy.dtype_name == dtype_name
        if strategy_name == "none":
            assert strategy.container is None
        else:
            assert strategy.container.dtype_name == dtype_name

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("r_dtype, dtype_name", [(torch.float32, "float32"), (torch.float64, "float64")])
    def test_device_gives_its_precision_to_its_strategy(self, strategy_name: str, r_dtype, dtype_name: str):
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name, r_dtype=r_dtype)

        assert device.contraction_strategy.dtype_name == dtype_name
        if strategy_name != "none":
            assert device.contraction_strategy.container.dtype_name == dtype_name

    @pytest.mark.parametrize("strategy_name", [name for name in STRATEGIES if name != "bulk"])
    def test_a_strategy_without_plan_returns_none(self, strategy_name: str, rng: np.random.Generator):
        records = SptmCircuitRecords()
        records.add_gate(SptmCompRyRy, 0, 4, rng.uniform(size=2))
        records.add_blocks([(4, np.eye(4))])

        assert get_contraction_strategy(strategy_name).build_plan(records, 2 * N_WIRES, "float64") is None
        assert len(records) == 2

    @pytest.mark.parametrize(
        "make_operation",
        [
            lambda: SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]),
            lambda: SptmCompZX(wires=[0, 1]),
            lambda: SingleParticleTransitionMatrixOperation(np.eye(4, dtype=np.float32), wires=[0, 1]),
            lambda: CompRyRy(np.array([0.3, 0.4]), wires=[0, 1]),
        ],
        ids=["rotation", "swap", "plain_operation", "matchgate"],
    )
    def test_cast_without_dtype_name_returns_the_operation(self, make_operation):
        container = _ForwardMatchgatesContainer()
        operation = make_operation()

        assert container.dtype_name is None
        assert container.cast(operation) is operation
        if isinstance(operation, SptmCompRyRy):
            assert operation._stored_matrix is None

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_cast_returns_the_operation_when_the_dtype_matches(self, backend: str, dtype_name: str):
        container = _ForwardMatchgatesContainer()
        container.dtype_name = dtype_name
        matrix = np.eye(4, dtype=dtype_name)
        angles = np.array([0.3, 0.4], dtype=dtype_name)
        plain = SingleParticleTransitionMatrixOperation(
            torch.from_numpy(matrix) if backend == "torch" else matrix, wires=[0, 1]
        )
        gate = SptmCompRzRz(torch.from_numpy(angles) if backend == "torch" else angles, wires=[1, 2])

        assert container.cast(plain) is plain
        assert container.cast(gate) is gate

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("stored_dtype, dtype_name", [("float64", "float32"), ("float32", "float64")])
    @pytest.mark.parametrize("gate_name", list(GATE_CLASSES))
    def test_cast_converts_a_gate_to_a_plain_operation_of_the_dtype(
        self, backend: str, stored_dtype: str, dtype_name: str, gate_name: str, rng: np.random.Generator
    ):
        container = _ForwardMatchgatesContainer()
        container.dtype_name = dtype_name
        angles = rng.uniform(-np.pi, np.pi, size=2).astype(stored_dtype)
        gate = GATE_CLASSES[gate_name](
            torch.from_numpy(angles) if backend == "torch" else angles, wires=[1, 2], check_real=False
        )

        cast_operation = container.cast(gate)

        assert type(cast_operation) is SingleParticleTransitionMatrixOperation
        assert cast_operation.wires == gate.wires
        assert cast_operation.hyperparameters == gate.hyperparameters
        assert cast_operation.dtype == dtype_name
        assert isinstance(cast_operation.sptm_matrix(), torch.Tensor if backend == "torch" else np.ndarray)
        np.testing.assert_allclose(
            self._to_numpy(cast_operation.sptm_matrix()),
            GATE_CLASSES[gate_name].compute_sptm(angles.astype(np.float64)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_cast_converts_an_integer_swap(self):
        container = _ForwardMatchgatesContainer()
        container.dtype_name = "float32"
        swap = SptmCompZX(wires=[0, 2])

        cast_operation = container.cast(swap)

        assert cast_operation.dtype == "float32"
        assert cast_operation.wires == swap.wires
        np.testing.assert_array_equal(cast_operation.sptm_matrix(), np.asarray(swap.sptm_matrix()))

    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_cast_converts_a_matchgate_to_its_transition_matrix(self, dtype_name: str):
        container = _ForwardMatchgatesContainer()
        container.dtype_name = dtype_name
        angles = np.array([0.3, 0.4])

        cast_operation = container.cast(CompRyRy(angles, wires=[0, 1]))

        assert isinstance(cast_operation, SingleParticleTransitionMatrixOperation)
        assert cast_operation.dtype == dtype_name
        np.testing.assert_allclose(
            self._to_numpy(cast_operation.sptm_matrix()),
            SptmCompRyRy.compute_sptm(angles),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_cast_keeps_the_gradient_of_the_angles(self):
        container = _ForwardMatchgatesContainer()
        container.dtype_name = "float32"
        angles = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)

        cast_operation = container.cast(SptmCompRyRy(angles, wires=[0, 1]))
        (gradient,) = torch.autograd.grad(cast_operation.sptm_matrix().sum(), angles)

        assert cast_operation.sptm_matrix().dtype == torch.float32
        assert bool(torch.any(gradient != 0))

    @pytest.mark.parametrize("dtype_name", [None, "float32"])
    def test_push_contract_adds_the_operation_in_the_dtype_of_the_container(self, dtype_name):
        container = _ForwardMatchgatesContainer()
        container.dtype_name = dtype_name
        operations = [
            SptmCompRyRy(np.array([0.3, 0.4]), wires=[0, 1]),
            SptmCompRzRz(np.array([0.5, 0.1]), wires=[2, 3]),
        ]

        for operation in operations:
            assert container.push_contract(operation) is None

        assert len(container) == 2
        for stored, operation in zip(container.values(), operations):
            assert stored.dtype == (dtype_name or "float64")
            assert (stored is operation) == (dtype_name is None)

    @pytest.mark.parametrize("strategy_name, backend", HANDED_OPERATIONS_CASES)
    def test_a_strategy_hands_over_operations_in_the_precision_of_the_device(
        self, strategy_name: str, backend: str, rng: np.random.Generator
    ):
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name, r_dtype=torch.float32)
        strategy = device.contraction_strategy
        handed: List[Any] = []

        for operation in self._make_operations(rng, backend):
            assert operation.dtype in ("float64", "int64")
            handed.extend(strategy.get_next_operations(operation))
            for stored in strategy.container.values():
                assert set(self._block_dtype_names(stored)) == {"float32"}
        handed.append(strategy.get_reminding())

        handed = [operation for operation in handed if operation is not None]
        assert len(handed) > 0
        for operation in handed:
            assert set(self._block_dtype_names(operation)) == {"float32"}

    @pytest.mark.parametrize("strategy_name", ["bulk", "none"])
    def test_a_strategy_that_does_not_contract_hands_over_the_gates_as_they_are(
        self, strategy_name: str, rng: np.random.Generator
    ):
        device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name, r_dtype=torch.float32)

        for operation in self._make_operations(rng, "numpy"):
            (handed,) = device.contraction_strategy.get_next_operations(operation)
            assert handed is operation
        assert device.contraction_strategy.get_reminding() is None

    @pytest.mark.parametrize("strategy_name", STRATEGIES)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_single_precision_device_agrees_with_the_double_precision_one(
        self, strategy_name: str, backend: str, rng: np.random.Generator
    ):
        operations = self._make_operations(rng, backend)
        matrices = {}
        for r_dtype in (torch.float32, torch.float64):
            device = NonInteractingFermionicDevice(wires=N_WIRES, contraction_strategy=strategy_name, r_dtype=r_dtype)
            device.apply(operations)
            matrices[r_dtype] = device.global_sptm.sptm_matrix()
            assert str(matrices[r_dtype].dtype).replace("torch.", "") == str(r_dtype).replace("torch.", "")
            assert isinstance(matrices[r_dtype], torch.Tensor if backend == "torch" else np.ndarray)

        for operation in operations:
            assert operation.dtype in ("float64", "int64")
        np.testing.assert_allclose(
            self._to_numpy(matrices[torch.float32]),
            self._to_numpy(matrices[torch.float64]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_a_strategy_instance_is_accepted_by_the_device(self):
        strategy = NeighboursContractionStrategy()
        device = NonInteractingFermionicDevice(wires=3, contraction_strategy=strategy, r_dtype=torch.float32)
        assert device.contraction_strategy is strategy
        assert strategy.dtype_name == "float32"
        assert get_contraction_strategy(strategy) is strategy

    def test_an_unknown_strategy_name_is_refused(self):
        with pytest.raises(ValueError, match="Unknown contraction strategy"):
            get_contraction_strategy("no such strategy")
