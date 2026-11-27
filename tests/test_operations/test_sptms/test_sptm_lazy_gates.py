import copy
from typing import Any, List

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.operations import CompRxRx, CompRyRy, CompRzRz
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
)

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

BATCH_SIZE = 3
ROTATIONS = {"rxrx": SptmCompRxRx, "ryry": SptmCompRyRy, "rzrz": SptmCompRzRz}
MATCHGATES = {"rxrx": CompRxRx, "ryry": CompRyRy, "rzrz": CompRzRz}
ANGLES = np.array([0.3, 1.1])
OTHER_ANGLES = np.array([0.7, 0.2])


@pytest.mark.parametrize("gate_name", list(ROTATIONS))
class TestSptmLazyGates:
    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _dense(gate: SingleParticleTransitionMatrixOperation) -> SingleParticleTransitionMatrixOperation:
        return SingleParticleTransitionMatrixOperation(
            np.array(type(gate).compute_sptm(gate._params)), wires=gate.wires, **gate.hyperparameters
        )

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    def test_matrix_is_not_computed_at_construction(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])

        assert gate._stored_matrix is None
        assert gate.wires == qml.wires.Wires([0, 1])

    def test_data_is_the_angles(self, gate_name: str):
        angles = ANGLES.copy()
        gate = ROTATIONS[gate_name](angles, wires=[0, 1])

        assert gate.num_params == 1
        assert len(gate.data) == 1
        np.testing.assert_array_equal(gate.data[0], angles)
        np.testing.assert_array_equal(gate.parameters[0], angles)
        gate.sptm_matrix()
        np.testing.assert_array_equal(gate.data[0], angles)

    def test_matrix_is_computed_once(self, gate_name: str, monkeypatch: pytest.MonkeyPatch):
        gate_class = ROTATIONS[gate_name]
        formula = gate_class.compute_sptm
        calls: List[Any] = []

        def counting_formula(params: Any) -> Any:
            calls.append(params)
            return formula(params)

        monkeypatch.setattr(gate_class, "compute_sptm", staticmethod(counting_formula))
        gate = gate_class(ANGLES.copy(), wires=[0, 1])
        assert len(calls) == 0

        first: Any = gate.sptm_matrix()
        second = gate.sptm_matrix()
        third = gate.single_particle_transition_matrix

        assert len(calls) == 1
        assert first is second and second is third
        assert gate._stored_matrix is first
        np.testing.assert_allclose(
            np.asarray(first), formula(ANGLES), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize(
        "angles_shape, expected_batch_size",
        [((2,), None), ((1, 2), 1), ((BATCH_SIZE, 2), BATCH_SIZE)],
    )
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_batch_size_is_the_same_before_and_after_the_matrix_is_computed(
        self, gate_name: str, angles_shape, expected_batch_size, backend: str, rng: np.random.Generator
    ):
        angles = rng.uniform(0.1, 3.0, size=angles_shape)
        gate = ROTATIONS[gate_name](torch.from_numpy(angles) if backend == "torch" else angles, wires=[0, 1])

        assert gate.batch_size == expected_batch_size
        assert gate._stored_matrix is None
        matrix = gate.sptm_matrix()
        assert gate._stored_matrix is not None
        assert gate.batch_size == expected_batch_size
        assert tuple(matrix.shape) == (*angles_shape[:-1], 4, 4)
        assert gate.shape == (*angles_shape[:-1], 4, 4)

    def test_check_matrix_accepts_a_rotation(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1], check_matrix=True)

        assert gate.hyperparameters["check_matrix"]
        assert gate.check_is_in_so4()
        np.testing.assert_allclose(
            gate.sptm_matrix(),
            ROTATIONS[gate_name].compute_sptm(ANGLES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_check_matrix_refuses_a_matrix_out_of_so4(self, gate_name: str, monkeypatch: pytest.MonkeyPatch):
        gate_class = ROTATIONS[gate_name]
        monkeypatch.setattr(gate_class, "compute_sptm", staticmethod(lambda params: 2.0 * np.eye(4)))

        with pytest.raises(ValueError, match="not in SO"):
            gate_class(ANGLES.copy(), wires=[0, 1], check_matrix=True)

    def test_hyperparameters_are_kept(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1], check_real=False, normalize=True, extra_option=3)

        assert set(gate.hyperparameters) == {"clip_angles", "check_angles", "check_matrix", "check_real"}
        assert not gate.hyperparameters["check_real"]
        assert gate.kwargs == {"extra_option": 3}

    def test_matrix_can_be_replaced(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        replacement = np.eye(4)

        gate._matrix = replacement

        assert gate.sptm_matrix() is replacement
        assert gate.batch_size is None

    @pytest.mark.parametrize("wires", [[0, 1], [2, 3], [1, 0]])
    def test_decomposition_is_that_of_the_dense_operation(self, gate_name: str, wires):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=wires)
        dense = self._dense(gate)

        with qml.queuing.AnnotatedQueue() as queue:
            decomposition = gate.decomposition()

        assert len(decomposition) == 1
        assert isinstance(decomposition[0], qml.QubitUnitary)
        assert decomposition[0].wires == gate.wires
        assert queue.queue == decomposition
        np.testing.assert_allclose(
            qml.matrix(decomposition[0]),
            qml.matrix(dense.decomposition()[0]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_qubit_matrix_is_that_of_the_dense_operation_and_of_the_matchgate(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        matchgate = MATCHGATES[gate_name](ANGLES.copy(), wires=[0, 1])

        gate_matrix = qml.matrix(gate)

        np.testing.assert_allclose(
            gate_matrix, qml.matrix(self._dense(gate)), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        # the qubit unitary of a transition matrix is defined up to a sign
        overlap = np.trace(np.conj(np.asarray(matchgate.matrix())).T @ gate_matrix) / 4
        np.testing.assert_allclose(np.abs(overlap), 1.0, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("computed_before_copy", [False, True])
    def test_shallow_copy_gives_the_same_gate(self, gate_name: str, computed_before_copy: bool):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        if computed_before_copy:
            gate.sptm_matrix()

        duplicate = copy.copy(gate)

        assert type(duplicate) is type(gate)
        assert duplicate is not gate
        assert duplicate.wires == gate.wires
        assert (duplicate._stored_matrix is None) == (not computed_before_copy)
        np.testing.assert_array_equal(duplicate.data[0], ANGLES)
        np.testing.assert_allclose(
            duplicate.sptm_matrix(),
            ROTATIONS[gate_name].compute_sptm(ANGLES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        assert qml.equal(duplicate, gate)

    def test_deep_copy_gives_the_same_gate(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])

        duplicate = copy.deepcopy(gate)

        assert duplicate._params is not gate._params
        np.testing.assert_allclose(
            duplicate.sptm_matrix(), gate.sptm_matrix(), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("computed_before_map", [False, True])
    def test_map_wires_moves_the_gate(self, gate_name: str, computed_before_map: bool):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        if computed_before_map:
            gate.sptm_matrix()

        mapped = gate.map_wires({0: 2, 1: 3})

        assert type(mapped) is type(gate)
        assert mapped.wires == qml.wires.Wires([2, 3])
        assert mapped.cs_wires == qml.wires.Wires([2, 3])
        assert gate.wires == qml.wires.Wires([0, 1])
        expected = np.eye(8)
        expected[4:, 4:] = ROTATIONS[gate_name].compute_sptm(ANGLES)
        np.testing.assert_allclose(
            mapped.sptm_matrix(range(4)), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_gates_with_the_same_angles_are_equal(self, gate_name: str):
        first = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        second = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])

        assert qml.equal(first, second)
        assert first._stored_matrix is None and second._stored_matrix is None

    @pytest.mark.parametrize("computed", [False, True])
    def test_gates_with_other_angles_or_wires_are_not_equal(self, gate_name: str, computed: bool):
        gate_class = ROTATIONS[gate_name]
        gate = gate_class(ANGLES.copy(), wires=[0, 1])
        other_angles = gate_class(OTHER_ANGLES.copy(), wires=[0, 1])
        other_wires = gate_class(ANGLES.copy(), wires=[1, 2])
        if computed:
            for operation in (gate, other_angles, other_wires):
                operation.sptm_matrix()

        assert not qml.equal(gate, other_angles)
        assert not qml.equal(gate, other_wires)

    def test_gates_of_two_classes_are_not_equal(self, gate_name: str):
        other_name = {"rxrx": "ryry", "ryry": "rzrz", "rzrz": "rxrx"}[gate_name]
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        other = ROTATIONS[other_name](ANGLES.copy(), wires=[0, 1])

        assert not qml.equal(gate, other)

    def test_a_plain_operation_with_the_same_matrix_is_equal_to_the_gate(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        dense = self._dense(gate)

        assert qml.equal(dense, gate)
        qml.assert_equal(dense, gate)
        # PennyLane requires its second argument to be an instance of the type of the first
        assert not qml.equal(gate, dense)

    def test_a_plain_operation_that_differs_is_not_equal_to_the_gate(self, gate_name: str):
        gate_class = ROTATIONS[gate_name]
        gate = gate_class(ANGLES.copy(), wires=[0, 1])
        dense = self._dense(gate)
        torch_gate = gate_class(torch.from_numpy(ANGLES.copy()), wires=[0, 1])
        trainable_gate = gate_class(torch.from_numpy(ANGLES.copy()).requires_grad_(), wires=[0, 1])
        torch_dense = SingleParticleTransitionMatrixOperation(
            torch.from_numpy(np.array(dense.sptm_matrix())), wires=[0, 1], **gate.hyperparameters
        )
        other_hyperparameters = SingleParticleTransitionMatrixOperation(
            np.array(dense.sptm_matrix()), wires=[0, 1], **{**gate.hyperparameters, "check_real": False}
        )

        assert not qml.equal(dense, gate_class(OTHER_ANGLES.copy(), wires=[0, 1]))
        assert not qml.equal(dense, gate_class(ANGLES.copy(), wires=[1, 2]))
        assert not qml.equal(dense, gate_class(ANGLES[None].copy(), wires=[0, 1]))
        assert not qml.equal(other_hyperparameters, gate)
        assert not qml.equal(dense, torch_gate)
        assert qml.equal(dense, torch_gate, check_interface=False)
        assert qml.equal(torch_dense, torch_gate)
        assert not qml.equal(torch_dense, trainable_gate)
        assert qml.equal(torch_dense, trainable_gate, check_trainability=False)

    def test_bind_new_parameters_gives_the_gate_of_the_new_angles(self, gate_name: str):
        gate_class = ROTATIONS[gate_name]
        gate = gate_class(ANGLES.copy(), wires=[2, 3])
        gate.sptm_matrix()

        bound = qml.ops.functions.bind_new_parameters(gate, [OTHER_ANGLES.copy()])

        assert type(bound) is gate_class
        assert bound.wires == gate.wires
        assert bound.hyperparameters == gate.hyperparameters
        np.testing.assert_array_equal(bound.data[0], OTHER_ANGLES)
        np.testing.assert_allclose(
            bound.sptm_matrix(),
            gate_class.compute_sptm(OTHER_ANGLES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            gate.sptm_matrix(),
            gate_class.compute_sptm(ANGLES),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("angles_shape", [(2,), (BATCH_SIZE, 2)])
    def test_adjoint_is_the_transpose(self, gate_name: str, angles_shape, rng: np.random.Generator):
        angles = rng.uniform(0.1, 3.0, size=angles_shape)
        gate = ROTATIONS[gate_name](angles, wires=[0, 1])
        expected = np.swapaxes(ROTATIONS[gate_name].compute_sptm(angles), -2, -1)

        adjoint = gate.adjoint()
        lazy_adjoint = SingleParticleTransitionMatrixOperation.from_operation(qml.adjoint(gate))

        assert adjoint.wires == gate.wires
        for operation in (adjoint, lazy_adjoint):
            np.testing.assert_allclose(
                operation.sptm_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    @pytest.mark.parametrize("angles_shape", [(2,), (BATCH_SIZE, 2)])
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_pad_embeds_the_matrix_in_the_identity(
        self, gate_name: str, angles_shape, backend: str, rng: np.random.Generator
    ):
        angles = rng.uniform(0.1, 3.0, size=angles_shape)
        gate = ROTATIONS[gate_name](torch.from_numpy(angles) if backend == "torch" else angles, wires=[1, 2])
        expected = np.zeros((*angles_shape[:-1], 8, 8))
        expected[...] = np.eye(8)
        expected[..., 2:6, 2:6] = ROTATIONS[gate_name].compute_sptm(angles)

        padded = gate.pad(range(4))

        assert padded.wires == qml.wires.Wires(range(4))
        np.testing.assert_allclose(
            self._to_numpy(padded.sptm_matrix()), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert gate.pad([1, 2]) is gate

    def test_to_matchgate_gives_the_matchgate_of_the_angles(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])

        matchgate = gate.to_matchgate()

        assert isinstance(matchgate, MATCHGATES[gate_name])
        assert matchgate.wires == gate.wires
        assert gate._stored_matrix is None
        np.testing.assert_allclose(
            np.real(self._to_numpy(matchgate.single_particle_transition_matrix)),
            gate.sptm_matrix(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_matrix_of_torch_angles_is_differentiable(self, gate_name: str):
        angles = torch.from_numpy(ANGLES.copy()).requires_grad_()
        gate = ROTATIONS[gate_name](angles, wires=[0, 1])
        weights = torch.arange(16, dtype=torch.float64).reshape(4, 4)

        matrix = gate.sptm_matrix()
        (gradient,) = torch.autograd.grad((matrix * weights).sum(), angles)
        (expected,) = torch.autograd.grad((ROTATIONS[gate_name].compute_sptm(angles) * weights).sum(), angles)

        assert isinstance(matrix, torch.Tensor)
        assert gate.data[0] is angles
        torch.testing.assert_close(gradient, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        assert bool(torch.any(gradient != 0))

    def test_matmul_with_a_plain_operation(self, gate_name: str):
        gate = ROTATIONS[gate_name](ANGLES.copy(), wires=[0, 1])
        other = ROTATIONS[gate_name](OTHER_ANGLES.copy(), wires=[1, 2])
        expected = gate.pad(range(3)).sptm_matrix() @ other.pad(range(3)).sptm_matrix()

        product = gate @ other

        np.testing.assert_allclose(
            product.sptm_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_complex_angles_with_an_imaginary_part_are_refused_when_the_gate_is_built(self, gate_name: str):
        with pytest.raises(ValueError, match="must be real"):
            ROTATIONS[gate_name](np.array([0.3 + 0.2j, 0.5]), wires=[0, 1])

    def test_complex_angles_without_an_imaginary_part_give_a_real_matrix(self, gate_name: str):
        gate = ROTATIONS[gate_name](np.array([0.3 + 0.0j, 0.5 + 0.0j]), wires=[0, 1])
        assert gate._stored_matrix is not None
        assert not np.iscomplexobj(gate.sptm_matrix())
