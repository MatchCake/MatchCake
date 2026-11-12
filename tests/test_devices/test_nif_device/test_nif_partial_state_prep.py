import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake.operations.state_preparation import ProductState

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_WIRES = 4


class TestNonInteractingFermionicDevicePartialStatePrep:
    @staticmethod
    def gates():
        qml.IsingXY(0.5, wires=[0, 1])
        qml.IsingXY(0.9, wires=[1, 2])
        mc.operations.CompRxRx(np.array([0.3, -0.7]), wires=[2, 3])
        qml.IsingXY(-0.4, wires=[0, 1])

    @staticmethod
    def random_amplitudes(n_wires, batch_size=None):
        shape = ([batch_size] if batch_size is not None else []) + [n_wires, 2]
        amplitudes = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        return amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)

    @staticmethod
    def run(device_name, prepare, measurement, shots=None):
        @qml.set_shots(shots)
        @qml.qnode(qml.device(device_name, wires=N_WIRES))
        def circuit():
            prepare()
            TestNonInteractingFermionicDevicePartialStatePrep.gates()
            return measurement()

        return qml.math.to_numpy(circuit())

    def setup_method(self):
        set_seed(TEST_SEED)

    def assert_matches_default_qubit(self, prepare, measurement):
        np.testing.assert_allclose(
            self.run("nif.qubit", prepare, measurement),
            self.run("default.qubit", prepare, measurement),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "bits, wires",
        [
            ([1, 1], [1, 2]),
            ([1], [3]),
            ([1, 0, 1], [0, 2, 3]),
            ([1, 0, 0, 0], [3, 2, 1, 0]),
            ([0, 1, 1], [3, 0, 1]),
        ],
    )
    def test_basis_state_probs(self, bits, wires):
        self.assert_matches_default_qubit(
            lambda: qml.BasisState(np.array(bits), wires=wires),
            lambda: qml.probs(wires=range(N_WIRES)),
        )

    def test_basis_state_without_gates(self):
        def circuit():
            qml.BasisState(np.array([1, 1]), wires=[1, 2])
            return qml.probs(wires=range(N_WIRES))

        expected = np.zeros(2**N_WIRES)
        expected[0b0110] = 1.0
        np.testing.assert_allclose(
            qml.math.to_numpy(qml.QNode(circuit, qml.device("nif.qubit", wires=N_WIRES))()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
        )

    def test_basis_embedding_probs(self):
        self.assert_matches_default_qubit(
            lambda: qml.BasisEmbedding(np.array([1, 0, 1]), wires=[3, 1, 0]),
            lambda: qml.probs(wires=range(N_WIRES)),
        )

    @pytest.mark.parametrize("wires", [[1, 2], [2, 0], [3, 1, 0], [3, 2, 1, 0]])
    def test_product_state_probs(self, wires):
        amplitudes = self.random_amplitudes(len(wires))
        self.assert_matches_default_qubit(
            lambda: ProductState(amplitudes, wires=wires),
            lambda: qml.probs(wires=range(N_WIRES)),
        )

    @pytest.mark.parametrize(
        "make_observable",
        [lambda: qml.Z(0) @ qml.Z(3), lambda: qml.X(1) @ qml.X(2), lambda: qml.Y(0) @ qml.Y(1)],
    )
    def test_product_state_expval(self, make_observable):
        amplitudes = self.random_amplitudes(2)
        self.assert_matches_default_qubit(
            lambda: ProductState(amplitudes, wires=[2, 0]),
            lambda: qml.expval(make_observable()),
        )

    def test_batched_product_state_probs(self):
        amplitudes = self.random_amplitudes(2, batch_size=3)
        self.assert_matches_default_qubit(
            lambda: ProductState(amplitudes, wires=[3, 1]),
            lambda: qml.probs(wires=range(N_WIRES)),
        )

    def test_basis_state_expval_from_shots(self):
        def prepare():
            qml.BasisState(np.array([1, 1]), wires=[2, 0])

        def measurement():
            return qml.expval(qml.Z(0) @ qml.Z(1))

        np.testing.assert_allclose(
            self.run("nif.qubit", prepare, measurement, shots=4000),
            self.run("default.qubit", prepare, measurement),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_product_state_gradient_matches_default_qubit(self):
        def expval_and_grad(device_name):
            angles = torch.tensor([0.4, 1.1], dtype=torch.float64, requires_grad=True)

            @qml.qnode(qml.device(device_name, wires=N_WIRES), interface="torch")
            def circuit(angles):
                amplitudes = torch.stack([torch.cos(angles / 2), torch.sin(angles / 2)], dim=-1)
                ProductState(amplitudes.to(torch.complex128), wires=[3, 1])
                self.gates()
                return qml.expval(qml.Z(0) @ qml.Z(2))

            value = circuit(angles)
            value.backward()
            return value.detach().numpy(), angles.grad.numpy()

        nif_value, nif_grad = expval_and_grad("nif.qubit")
        qubit_value, qubit_grad = expval_and_grad("default.qubit")

        np.testing.assert_allclose(nif_value, qubit_value, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(nif_grad, qubit_grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_state_prep_on_device_wires_is_kept(self):
        device = mc.NIFDevice(wires=N_WIRES)
        state_prep = ProductState(self.random_amplitudes(N_WIRES), wires=range(N_WIRES))

        device.apply_state_prep(state_prep)

        assert device.state_prep_op is state_prep
