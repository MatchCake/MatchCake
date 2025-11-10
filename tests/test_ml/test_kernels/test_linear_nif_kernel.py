import numpy as np
import pytest
import torch.nn

from matchcake.ml.kernels.linear_nif_kernel import LinearNIFKernel


@pytest.mark.parametrize(
    "n_qubits, encoder_activation, bias",
    [(n_qubits, act, bias) for n_qubits in [3, 4, 8] for act in ["Identity", "Tanh"] for bias in [True, False]],
)
class TestLinearNIFKernel:
    @pytest.fixture(scope="function")
    def kernel_instance(self, n_qubits, encoder_activation, bias):
        instance = LinearNIFKernel(n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation)
        return instance

    @pytest.fixture
    def x_train(self):
        return np.linspace(0.0, 0.05, num=80).reshape(10, 8)

    def test_fit(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        assert kernel_instance.x_train_ is x_train
        assert kernel_instance.is_fitted_

    def test_swap_test(self, kernel_instance):
        x_train = np.stack([np.linspace(0.0, 1.0, num=8) for _ in range(10)])
        kernel_instance.fit(x_train)
        kernel_matrix = kernel_instance(x_train)
        torch.testing.assert_close(kernel_matrix, torch.ones_like(kernel_matrix))

    def test_symmetric_kernel(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        kernel_matrix = kernel_instance(x_train)
        torch.testing.assert_close(kernel_matrix, kernel_matrix.T)

    def test_n_qubits_setter(self, kernel_instance):
        n_qubits = kernel_instance.n_qubits
        kernel_instance.n_qubits = n_qubits + 1
        assert kernel_instance.n_qubits == n_qubits + 1
        assert kernel_instance.encoder[1].out_features == kernel_instance.encoder_out_indices[0].size

    def test_bias_setter(self, kernel_instance):
        new_bias = not kernel_instance.bias
        kernel_instance.bias = new_bias
        assert kernel_instance.bias == new_bias

    def test_encoder_activation_setter(self, kernel_instance):
        new_activation = "ReLU"
        kernel_instance.encoder_activation = new_activation
        assert kernel_instance.encoder_activation == new_activation
