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

    def test_custom_dtypes_propagate_to_device(self, n_qubits, encoder_activation, bias):
        kernel = LinearNIFKernel(
            n_qubits=n_qubits,
            bias=bias,
            encoder_activation=encoder_activation,
            r_dtype=torch.float64,
            c_dtype=torch.complex128,
        )
        assert kernel.R_DTYPE == torch.float64
        assert kernel.C_DTYPE == torch.complex128
        assert kernel.q_device.R_DTYPE == torch.float64
        assert kernel.q_device.C_DTYPE == torch.complex128
        kernel.n_qubits = n_qubits + 1
        assert kernel.q_device.R_DTYPE == torch.float64
        assert kernel.q_device.C_DTYPE == torch.complex128

    def test_failed_refit_on_another_feature_count_keeps_the_encoder(self, kernel_instance, x_train):
        kernel_instance.alignment = True
        kernel_instance.alignment_iterations = 2
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        linear = kernel_instance.encoder[1]
        with pytest.raises(Exception):
            kernel_instance.fit(np.concatenate([x_train, x_train[:, :4]], axis=-1), np.arange(4) % 2)
        assert kernel_instance.x_train_ is x_train
        assert kernel_instance.encoder[1] is linear
        assert linear.in_features == 8
        assert not linear.training
        assert kernel_instance.transform(x_train).shape == (len(x_train), len(x_train))

    def fail_at_call(self, kernel_instance, n_calls):
        forward = kernel_instance.forward
        calls = []

        def failing_forward(*args, **kwargs):
            calls.append(1)
            if len(calls) >= n_calls:
                raise RuntimeError("forward failed")
            return forward(*args, **kwargs)

        kernel_instance.forward = failing_forward

    @pytest.mark.parametrize("r_dtype, c_dtype", [(torch.float32, torch.complex64), (torch.float64, torch.complex128)])
    def test_alignment_keeps_the_dtype_of_the_parameters(
        self, n_qubits, encoder_activation, bias, x_train, r_dtype, c_dtype
    ):
        kernel_instance = LinearNIFKernel(
            n_qubits=n_qubits,
            bias=bias,
            encoder_activation=encoder_activation,
            r_dtype=r_dtype,
            c_dtype=c_dtype,
            alignment=True,
            alignment_iterations=2,
        )
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        assert all(parameter.dtype == r_dtype for parameter in kernel_instance.parameters())

    def test_failed_alignment_keeps_the_trained_weights(self, kernel_instance, x_train):
        y_train = np.arange(len(x_train)) % 2
        kernel_instance.alignment = True
        kernel_instance.alignment_iterations = 4
        kernel_instance.fit(x_train, y_train)
        assert all(parameter.dtype == kernel_instance.R_DTYPE for parameter in kernel_instance.parameters())
        weights = [parameter.detach().clone() for parameter in kernel_instance.parameters()]
        expected = kernel_instance.transform(x_train)
        self.fail_at_call(kernel_instance, n_calls=4)
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel_instance.fit(x_train, y_train)
        del kernel_instance.forward
        for parameter, weight in zip(kernel_instance.parameters(), weights):
            torch.testing.assert_close(parameter.detach(), weight, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(kernel_instance.transform(x_train), expected)

    def test_fit_after_a_failed_first_fit_equals_a_first_fit(self, n_qubits, encoder_activation, bias, x_train):
        y_train = np.arange(len(x_train)) % 2
        kernel_arguments = dict(
            n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation, alignment=True, alignment_iterations=4
        )
        expected = LinearNIFKernel(**kernel_arguments).fit(x_train, y_train).transform(x_train)
        kernel_instance = LinearNIFKernel(**kernel_arguments)
        linear = kernel_instance.encoder[1]
        self.fail_at_call(kernel_instance, n_calls=4)
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel_instance.fit(x_train, y_train)
        del kernel_instance.forward
        assert kernel_instance.encoder[1] is linear
        assert not kernel_instance.is_fitted_
        assert any(torch.nn.parameter.is_lazy(parameter) for parameter in kernel_instance.parameters())
        np.testing.assert_array_equal(kernel_instance.fit(x_train, y_train).transform(x_train), expected)

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

    def test_encoder_weights_follow_random_state(self, n_qubits, encoder_activation, bias, x_train):
        kernels = []
        for global_seed, random_state in [(1, 0), (2, 0), (2, 1)]:
            torch.manual_seed(global_seed)
            kernel = LinearNIFKernel(
                n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation, random_state=random_state
            )
            kernels.append(kernel.fit(x_train))
        torch.testing.assert_close(kernels[0].encoder[1].weight, kernels[1].encoder[1].weight, atol=0.0, rtol=0.0)
        assert not torch.equal(kernels[1].encoder[1].weight, kernels[2].encoder[1].weight)

    def test_encoder_weights_follow_the_default_linear_distribution(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        linear = kernel_instance.encoder[1]
        bound = linear.in_features**-0.5
        generator = torch.Generator().manual_seed(kernel_instance.random_state)
        expected = torch.empty(linear.weight.shape, dtype=linear.weight.dtype).uniform_(
            -bound, bound, generator=generator
        )
        torch.testing.assert_close(linear.weight.detach(), expected, atol=0.0, rtol=0.0)
        assert linear.weight.abs().max() > 0.5 * bound

    def test_fit_without_random_state(self, n_qubits, encoder_activation, bias, x_train):
        kernel = LinearNIFKernel(n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation)
        kernel.random_state = None
        kernel.fit(x_train)
        assert kernel.is_fitted_

    def test_fit_does_not_initialize_cuda(self, kernel_instance, x_train):
        was_initialized = torch.cuda.is_initialized()
        kernel_instance.fit(x_train)
        assert torch.cuda.is_initialized() == was_initialized

    def test_refit_keeps_the_encoder_weights(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        with torch.no_grad():
            kernel_instance.encoder[1].weight.add_(0.5)
        weights = kernel_instance.encoder[1].weight.detach().clone()
        kernel_instance.fit(x_train)
        torch.testing.assert_close(kernel_instance.encoder[1].weight, weights, atol=0.0, rtol=0.0)

    def test_transform_follows_the_state_of_the_kernel(self, kernel_instance, x_train):
        x_test = torch.from_numpy(x_train[:4] + 0.01)

        def assert_transform_is_up_to_date():
            transformed = kernel_instance.transform(x_test)
            with torch.no_grad():
                expected = kernel_instance(x_test, kernel_instance.x_train_)
            torch.testing.assert_close(transformed, expected)

        kernel_instance.fit(x_train)
        assert_transform_is_up_to_date()
        with torch.no_grad():
            kernel_instance.encoder[1].weight.add_(0.05)
        assert_transform_is_up_to_date()
        kernel_instance.encoder_activation = "Sigmoid"
        assert_transform_is_up_to_date()
        kernel_instance.encoder[-1] = torch.nn.LeakyReLU()
        assert_transform_is_up_to_date()
        kernel_instance.encoder[-1].negative_slope = 0.5
        assert_transform_is_up_to_date()
        kernel_instance.bias = not kernel_instance.bias
        assert_transform_is_up_to_date()
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        assert_transform_is_up_to_date()
        kernel_instance.alignment_iterations = 2
        kernel_instance._align_kernel()
        assert_transform_is_up_to_date()

    def test_refit_with_another_number_of_features(self, n_qubits, encoder_activation, bias, x_train):
        kernel_instance = LinearNIFKernel(n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation)
        wide_x_train = np.concatenate([x_train, x_train[::-1]], axis=-1)
        kernel_instance.fit(x_train)
        kernel_instance.fit(wide_x_train)
        assert kernel_instance.encoder[1].in_features == 16
        fresh_instance = LinearNIFKernel(n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation)
        fresh_instance.fit(wide_x_train)
        torch.testing.assert_close(
            kernel_instance.encoder[1].weight, fresh_instance.encoder[1].weight, atol=0.0, rtol=0.0
        )
        x_test = torch.from_numpy(wide_x_train[:3] + 0.01)
        torch.testing.assert_close(kernel_instance.transform(x_test), fresh_instance.transform(x_test))

    def test_refit_with_fewer_features_of_another_shape(self, n_qubits, encoder_activation, bias, x_train):
        kernel_instance = LinearNIFKernel(n_qubits=n_qubits, bias=bias, encoder_activation=encoder_activation)
        kernel_instance.fit(np.concatenate([x_train, x_train[::-1]], axis=-1))
        kernel_instance.freeze()
        kernel_instance.fit(x_train.reshape(10, 2, 4))
        assert kernel_instance.encoder[1].in_features == 8
        assert not kernel_instance.encoder[1].weight.requires_grad
        assert not kernel_instance.encoder[1].training
        kernel_instance.fit(np.stack([x_train, x_train], axis=1))
        assert kernel_instance.encoder[1].in_features == 16

    def test_fit_with_a_custom_encoder(self, kernel_instance, x_train):
        n_outputs = kernel_instance.encoder_out_indices[0].size
        encoders = [
            torch.nn.Linear(8, n_outputs),
            torch.nn.Sequential(torch.nn.Linear(8, n_outputs)),
            torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Tanh(), torch.nn.Linear(8, n_outputs)),
        ]
        for encoder in encoders:
            kernel_instance.encoder = encoder
            weights = [parameter.detach().clone() for parameter in encoder.parameters()]
            kernel_instance.fit(x_train)
            assert kernel_instance.transform(torch.from_numpy(x_train)).shape == (10, 10)
            for parameter, weight in zip(encoder.parameters(), weights):
                torch.testing.assert_close(parameter, weight, atol=0.0, rtol=0.0)

    def test_float64_kernel_and_alignment(self, n_qubits, encoder_activation, bias, x_train):
        kernel_instance = LinearNIFKernel(
            n_qubits=n_qubits,
            bias=bias,
            encoder_activation=encoder_activation,
            r_dtype=torch.float64,
            c_dtype=torch.complex128,
            alignment=True,
            alignment_iterations=2,
        )
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        assert len(kernel_instance.opt_.history) == 2
        assert kernel_instance.transform(torch.from_numpy(x_train)).dtype == torch.float64

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
