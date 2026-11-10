import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_WIRES = 3


def random_hermitian(n_wires: int, seed: int, is_complex: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimension = 2**n_wires
    matrix = rng.normal(size=(dimension, dimension))
    if is_complex:
        matrix = matrix + 1j * rng.normal(size=(dimension, dimension))
    return matrix + matrix.conj().T


class TestNonInteractingFermionicDeviceHermitianExpval:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @staticmethod
    def circuit(theta, matrix, wires):
        qml.BasisState(np.array([0, 1, 1]), wires=range(N_WIRES))
        qml.SingleExcitation(theta, wires=[0, 1])
        qml.SingleExcitation(-0.4, wires=[1, 2])
        return qml.expval(qml.Hermitian(matrix, wires=wires))

    @pytest.mark.parametrize(
        "wires, is_complex",
        [
            ([0], False),
            ([1], True),
            ([0, 1], True),
            ([2, 0], True),
            ([1, 2, 0], True),
            ([0, 1, 2], False),
        ],
    )
    def test_matches_default_qubit(self, wires, is_complex):
        matrix = random_hermitian(len(wires), seed=len(wires), is_complex=is_complex)
        expected = qml.QNode(self.circuit, qml.device("default.qubit", wires=N_WIRES))(0.8, matrix, wires)
        result = qml.QNode(self.circuit, NonInteractingFermionicDevice(wires=N_WIRES))(0.8, matrix, wires)
        np.testing.assert_allclose(
            float(result), float(expected), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.diag([1.0, -1.0]), np.eye(4)])
    def test_sparse_matrices(self, matrix):
        wires = list(range(int(np.log2(matrix.shape[0]))))
        expected = qml.QNode(self.circuit, qml.device("default.qubit", wires=N_WIRES))(0.8, matrix, wires)
        result = qml.QNode(self.circuit, NonInteractingFermionicDevice(wires=N_WIRES))(0.8, matrix, wires)
        np.testing.assert_allclose(
            float(result), float(expected), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_batched_circuit(self):
        theta = np.array([0.8, 0.3, -1.2])
        matrix = random_hermitian(2, seed=7)
        expected = qml.QNode(self.circuit, qml.device("default.qubit", wires=N_WIRES))(theta, matrix, [2, 0])
        result = qml.QNode(self.circuit, NonInteractingFermionicDevice(wires=N_WIRES))(theta, matrix, [2, 0])
        np.testing.assert_allclose(
            np.asarray(result), np.asarray(expected), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize(
        "matrix",
        [
            random_hermitian(2, seed=3, is_complex=False),
            random_hermitian(2, seed=5),
            np.zeros((4, 4)),
        ],
    )
    def test_gradient_matches_default_qubit(self, matrix):
        dtype = torch.complex128 if np.iscomplexobj(matrix) else torch.float64
        theta = torch.tensor(0.8, dtype=torch.float64)
        gradients = []
        for device in [qml.device("default.qubit", wires=N_WIRES), NonInteractingFermionicDevice(wires=N_WIRES)]:
            trainable_matrix = torch.tensor(matrix, dtype=dtype, requires_grad=True)
            qml.QNode(self.circuit, device, interface="torch")(theta, trainable_matrix, [1, 2]).backward()
            gradients.append(trainable_matrix.grad)
        torch.testing.assert_close(gradients[1], gradients[0], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_gradcheck(self):
        theta = torch.tensor(0.8, dtype=torch.float64)
        device = NonInteractingFermionicDevice(wires=N_WIRES)
        qnode = qml.QNode(self.circuit, device, interface="torch")
        matrix = torch.tensor(random_hermitian(1, seed=11, is_complex=False), dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda trainable_matrix: qnode(theta, trainable_matrix, [0]), (matrix,))
