from typing import Any, Tuple

import numpy as np
import pytest
import torch

from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmCompRyRy,
    SptmCompRzRz,
    SptmCompZX,
)
from matchcake.utils import make_single_particle_transition_matrix_from_gate

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)

# name -> (gate class with a formula, matchgate it is the transition matrix of)
ROTATIONS = {
    "rxrx": (SptmCompRxRx, CompRxRx),
    "ryry": (SptmCompRyRy, CompRyRy),
    "rzrz": (SptmCompRzRz, CompRzRz),
}
ANGLES_SHAPES = [
    pytest.param((2,), id="one_gate"),
    pytest.param((5, 2), id="gates"),
    pytest.param((5, 3, 2), id="batched_gates"),
    pytest.param((1, 2), id="single_stacked_gate"),
]
SWAP_PARAMS_SHAPES = [
    pytest.param((0,), id="one_gate"),
    pytest.param((4, 0), id="gates"),
    pytest.param((4, 3, 0), id="batched_gates"),
]
SWAP_MATRIX = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)


class TestSptmComputeSptm:
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
    def _matchgate_matrices(matchgate_class: Any, angles: np.ndarray) -> np.ndarray:
        flat_angles = angles.reshape(-1, 2)
        matrices = [
            np.asarray(
                make_single_particle_transition_matrix_from_gate(
                    np.asarray(matchgate_class(gate_angles, wires=[0, 1]).matrix())
                )
            )
            for gate_angles in flat_angles
        ]
        return np.real(np.stack(matrices)).reshape(*angles.shape[:-1], 4, 4)

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(TEST_SEED)

    @pytest.mark.parametrize("gate_name", list(ROTATIONS))
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("angles_shape", ANGLES_SHAPES)
    def test_stacked_angles_give_the_matrix_of_each_matchgate(
        self, gate_name: str, backend: str, angles_shape: Tuple[int, ...], rng: np.random.Generator
    ):
        gate_class, matchgate_class = ROTATIONS[gate_name]
        angles = rng.uniform(0.1, 3.0, size=angles_shape)

        matrices = gate_class.compute_sptm(self._to_backend(angles, backend))

        assert isinstance(matrices, torch.Tensor if backend == "torch" else np.ndarray)
        assert tuple(matrices.shape) == (*angles_shape[:-1], 4, 4)
        np.testing.assert_allclose(
            self._to_numpy(matrices),
            self._matchgate_matrices(matchgate_class, angles),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("gate_name", list(ROTATIONS))
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("angles_shape", ANGLES_SHAPES)
    def test_stacked_angles_give_the_matrix_of_each_gate(
        self, gate_name: str, backend: str, angles_shape: Tuple[int, ...], rng: np.random.Generator
    ):
        gate_class, _ = ROTATIONS[gate_name]
        angles = rng.uniform(-np.pi, np.pi, size=angles_shape)
        expected = np.stack(
            [
                self._to_numpy(gate_class(self._to_backend(gate_angles, backend), wires=[0, 1]).sptm_matrix())
                for gate_angles in angles.reshape(-1, 2)
            ]
        ).reshape(*angles_shape[:-1], 4, 4)

        matrices = gate_class.compute_sptm(self._to_backend(angles, backend))

        np.testing.assert_allclose(
            self._to_numpy(matrices), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("gate_name", list(ROTATIONS))
    @pytest.mark.parametrize("angles_shape", ANGLES_SHAPES)
    def test_matrices_are_special_orthogonal(
        self, gate_name: str, angles_shape: Tuple[int, ...], rng: np.random.Generator
    ):
        gate_class, _ = ROTATIONS[gate_name]
        matrices = gate_class.compute_sptm(rng.uniform(-np.pi, np.pi, size=angles_shape))

        np.testing.assert_allclose(
            matrices @ np.swapaxes(matrices, -2, -1),
            np.broadcast_to(np.eye(4), matrices.shape),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            np.linalg.det(matrices),
            np.ones(angles_shape[:-1]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("gate_name", list(ROTATIONS))
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_dtype_follows_the_angles(self, gate_name: str, backend: str, dtype_name: str, rng: np.random.Generator):
        gate_class, _ = ROTATIONS[gate_name]
        angles = self._to_backend(rng.uniform(-np.pi, np.pi, size=(3, 2)).astype(dtype_name), backend)

        matrices = gate_class.compute_sptm(angles)

        assert str(matrices.dtype).replace("torch.", "") == dtype_name

    @pytest.mark.parametrize("gate_name", list(ROTATIONS))
    @pytest.mark.parametrize("angles_shape", ANGLES_SHAPES)
    def test_gradcheck(self, gate_name: str, angles_shape: Tuple[int, ...], rng: np.random.Generator):
        gate_class, _ = ROTATIONS[gate_name]
        angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=angles_shape)).requires_grad_()

        assert torch.autograd.gradcheck(
            gate_class.compute_sptm, (angles,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("params_shape", SWAP_PARAMS_SHAPES)
    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_swap_is_the_same_constant_for_every_gate(
        self, backend: str, params_shape: Tuple[int, ...], dtype_name: str
    ):
        params = self._to_backend(np.zeros(params_shape, dtype=dtype_name), backend)

        matrices = SptmCompZX.compute_sptm(params)

        assert isinstance(matrices, torch.Tensor if backend == "torch" else np.ndarray)
        assert tuple(matrices.shape) == (*params_shape[:-1], 4, 4)
        assert str(matrices.dtype).replace("torch.", "") == dtype_name
        np.testing.assert_array_equal(
            self._to_numpy(matrices), np.broadcast_to(SWAP_MATRIX, (*params_shape[:-1], 4, 4))
        )

    @pytest.mark.parametrize("wires", [[0, 1], [1, 0], [3, 4]])
    def test_swap_constant_is_the_matrix_of_the_gate_on_adjacent_wires(self, wires):
        gate = SptmCompZX(wires=wires)

        np.testing.assert_array_equal(SptmCompZX.compute_sptm(gate._params), np.asarray(gate.sptm_matrix()))

    def test_swap_constant_is_the_matrix_of_the_matchgate(self):
        expected = make_single_particle_transition_matrix_from_gate(np.asarray(fSWAP(wires=[0, 1]).matrix()))

        np.testing.assert_allclose(
            SptmCompZX.compute_sptm(np.zeros(0)),
            np.real(np.asarray(expected)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_swap_follows_the_gradient_free_parameters_of_torch(self):
        matrices = SptmCompZX.compute_sptm(torch.zeros((2, 0), dtype=torch.float64))

        assert matrices.device.type == "cpu"
        assert not matrices.requires_grad

    @pytest.mark.parametrize("gate_class", [SptmCompRxRx, SptmCompRyRy, SptmCompRzRz, SptmCompZX])
    def test_gate_classes_declare_their_formula(self, gate_class):
        assert gate_class.HAS_BULK_FORMULA

    def test_a_plain_operation_has_no_formula(self):
        assert not SingleParticleTransitionMatrixOperation.HAS_BULK_FORMULA
        with pytest.raises(NotImplementedError, match="no formula"):
            SingleParticleTransitionMatrixOperation.compute_sptm(np.zeros(2))
