from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from matchcake.ml.kernels.kernel import Kernel
from matchcake.utils.torch_utils import to_tensor


class TestKernel:
    @pytest.fixture
    def kernel(self):
        instance = Kernel()
        instance.forward = MagicMock()
        return instance

    def test_init(self, kernel):
        assert kernel.random_state == 0
        assert kernel.gram_batch_size == 10_000

    def test_transform(self, kernel):
        x = torch.rand(10, 10)
        kernel.transform(x)
        kernel.forward.assert_called_once()

    def test_fit(self, kernel):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10)
        kernel.fit(x, y)
        assert kernel.x_train_ is x
        assert kernel.y_train_ is y
        assert kernel.is_fitted_

    def test_failed_fit_keeps_the_state(self, kernel):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10)
        kernel.fit(x, y)
        kernel.forward = MagicMock(side_effect=RuntimeError("forward failed"))
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel.fit(torch.rand(5, 10), torch.rand(5, 10))
        assert kernel.x_train_ is x
        assert kernel.y_train_ is y
        assert kernel.is_fitted_

    @pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
    def test_failed_fit_restores_attributes_and_submodules(self, kernel, error_type):
        kernel.encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(10, 4))
        kernel.register_buffer("scale", torch.ones(1))
        kernel.eval()
        linear = kernel.encoder[1]
        linear.bias.requires_grad_(False)
        weight = linear.weight.detach().clone()
        hook_handle = linear.register_forward_hook(lambda *args: None)

        def failing_forward(*args, **kwargs):
            with torch.no_grad():
                linear.weight.add_(1.0)
                kernel.scale.add_(1.0)
            linear.requires_grad_(True)
            linear.register_forward_hook(lambda *args: None)
            kernel.register_forward_pre_hook(lambda *args: None)
            kernel.encoder[1] = torch.nn.Linear(5, 4)
            kernel.encoder.add_module("extra", torch.nn.Tanh())
            kernel.register_buffer("offset", torch.zeros(1))
            kernel.added_by_the_failed_fit = True
            kernel.train()
            raise error_type("forward failed")

        kernel.forward = failing_forward
        with pytest.raises(error_type, match="forward failed"):
            kernel.fit(torch.rand(5, 10))
        assert kernel.encoder[1] is linear
        assert len(kernel.encoder) == 2
        assert set(dict(kernel.named_buffers())) == {"scale"}
        assert not hasattr(kernel, "added_by_the_failed_fit")
        assert not kernel.training
        assert not kernel.encoder.training
        assert not linear.training
        torch.testing.assert_close(linear.weight.detach(), weight, atol=0.0, rtol=0.0)
        torch.testing.assert_close(kernel.scale, torch.ones(1), atol=0.0, rtol=0.0)
        assert linear.weight.requires_grad
        assert not linear.bias.requires_grad
        assert len(kernel._forward_pre_hooks) == 0
        assert len(linear._forward_hooks) == 1
        hook_handle.remove()
        assert len(linear._forward_hooks) == 0

    def test_failed_fit_leaves_lazy_layers_uninitialized(self, kernel):
        is_lazy = torch.nn.parameter.is_lazy
        kernel.encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(4), torch.nn.LazyBatchNorm1d())
        kernel.projection = torch.nn.LazyLinear(3)
        linear, batch_norm, projection = kernel.encoder[1], kernel.encoder[2], kernel.projection
        calls = []
        hook_handle = linear.register_forward_hook(lambda *args: calls.append(1))

        def failing_forward(x0, *args, **kwargs):
            kernel.projection(kernel.encoder(x0))
            raise RuntimeError("forward failed")

        kernel.forward = failing_forward
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel.fit(torch.rand(5, 10))
        assert kernel.encoder[1] is linear and kernel.encoder[2] is batch_norm and kernel.projection is projection
        assert isinstance(linear, torch.nn.LazyLinear)
        assert isinstance(batch_norm, torch.nn.LazyBatchNorm1d)
        assert is_lazy(linear.weight) and is_lazy(projection.weight)
        assert is_lazy(batch_norm.weight) and is_lazy(batch_norm.running_mean)
        assert len(calls) == 1
        assert kernel.projection(kernel.encoder(torch.rand(5, 7))).shape == (5, 3)
        assert linear.in_features == 7
        assert len(calls) == 2
        hook_handle.remove()
        kernel.encoder(torch.rand(5, 7))
        assert len(calls) == 2

    def test_fit_with_lazy_buffers(self, kernel):
        kernel.encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(4), torch.nn.LazyBatchNorm1d())
        kernel.forward = lambda x0, *args, **kwargs: kernel.encoder(x0)
        kernel.fit(torch.rand(5, 10))
        assert kernel.is_fitted_
        assert kernel.encoder[2].running_mean.shape == (4,)

    def test_failed_fit_restores_the_modules_of_an_encoder(self, kernel):
        kernel.encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(10, 4), torch.nn.BatchNorm1d(4))
        linear, batch_norm = kernel.encoder[1], kernel.encoder[2]
        buffers = {name: buffer.clone() for name, buffer in batch_norm.named_buffers()}

        def failing_forward(x0, *args, **kwargs):
            kernel.train()
            kernel.encoder(x0)
            kernel.register_buffer("temporary", torch.zeros(1), persistent=False)
            linear.register_full_backward_hook(lambda *args: None)
            kernel.double()
            raise RuntimeError("forward failed")

        kernel.forward = failing_forward
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel.fit(torch.rand(5, 10))
        for name, buffer in batch_norm.named_buffers():
            torch.testing.assert_close(buffer, buffers[name], atol=0.0, rtol=0.0)
        assert "temporary" not in kernel._non_persistent_buffers_set
        assert len(linear._backward_hooks) == 0
        assert linear.weight.dtype == torch.float32

    def test_failed_fit_in_inference_mode_keeps_trainable_parameters(self, kernel):
        kernel.encoder = torch.nn.Linear(10, 4)
        kernel.forward = MagicMock(side_effect=RuntimeError("forward failed"))
        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="forward failed"):
                kernel.fit(torch.rand(5, 10))
        assert not kernel.encoder.weight.is_inference()
        kernel.encoder(torch.rand(5, 10)).sum().backward()
        assert kernel.encoder.weight.grad is not None

    def test_failed_fit_raises_the_error_of_the_fit_when_a_restore_fails(self, kernel):
        class FrozenRegistry(dict):
            def clear(self):
                raise TypeError("frozen registry")

        kernel.encoder = torch.nn.Linear(10, 4)
        kernel._frozen_registry = FrozenRegistry()
        weight = kernel.encoder.weight.detach().clone()

        def failing_forward(*args, **kwargs):
            with torch.no_grad():
                kernel.encoder.weight.add_(1.0)
            raise RuntimeError("forward failed")

        kernel.forward = failing_forward
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel.fit(torch.rand(5, 10))
        torch.testing.assert_close(kernel.encoder.weight.detach(), weight, atol=0.0, rtol=0.0)

    def test_alignment_keeps_the_dtype_of_every_parameter(self, kernel):
        kernel.alignment = True
        kernel.alignment_iterations = 3
        kernel.scale = torch.nn.Parameter(torch.ones(10, dtype=torch.float32))
        kernel.shift = torch.nn.Parameter(torch.ones(10, dtype=torch.float64))
        kernel.count = torch.nn.Parameter(torch.ones(2, dtype=torch.long), requires_grad=False)

        def forward_mock(x0, x1=None):
            x0, x1 = to_tensor(x0), to_tensor(x0 if x1 is None else x1)
            return torch.einsum("j,ij,kj->ik", kernel.scale.double() * kernel.shift, x0, x1)

        kernel.forward = forward_mock
        kernel.fit(torch.rand(10, 10), torch.arange(10).long() % 2)
        assert kernel.scale.dtype == torch.float32
        assert kernel.shift.dtype == torch.float64
        assert kernel.count.dtype == torch.long
        assert not torch.equal(kernel.scale, torch.ones(10))

    def test_failed_first_fit_leaves_the_kernel_unfitted(self, kernel):
        kernel.forward = MagicMock(side_effect=RuntimeError("forward failed"))
        with pytest.raises(RuntimeError, match="forward failed"):
            kernel.fit(torch.rand(5, 10))
        assert kernel.x_train_ is None
        assert not kernel.is_fitted_

    def test_predict(self, kernel):
        x = torch.rand(10, 10)
        kernel.predict(x)
        kernel.forward.assert_called_once()

    def test_freeze(self, kernel):
        kernel.freeze()
        assert not kernel.training

    def test_alignment(self, kernel):
        kernel.alignment = True

        def forward_mock(x0, x1=None):
            if x1 is None:
                x1 = x0
            x0 = to_tensor(x0).float()
            x1 = to_tensor(x1).float()
            p = torch.ones_like(x0).float().requires_grad_()
            return torch.einsum("ij,ij,kj->ik", p, x0, x1)

        kernel.forward = forward_mock
        kernel.fit(torch.rand(10, 10), torch.arange(10).long())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_alignment_follows_the_dtype_of_the_kernel_matrix(self, kernel, dtype):
        kernel.alignment = True
        kernel.alignment_iterations = 2
        parameter = torch.nn.Parameter(torch.ones(10, dtype=dtype))
        kernel.scale = parameter

        def forward_mock(x0, x1=None):
            x0, x1 = to_tensor(x0).to(dtype), to_tensor(x0 if x1 is None else x1).to(dtype)
            return torch.einsum("j,ij,kj->ik", parameter, x0, x1)

        kernel.forward = forward_mock
        kernel.fit(torch.rand(10, 10), torch.arange(10).long() % 2)
        assert len(kernel.opt_.history) == 2
        assert all(np.isfinite(kernel.opt_.history))

    def test_alignment_with_regression_targets(self, kernel):
        kernel.alignment = True
        kernel.alignment_iterations = 2
        parameter = torch.nn.Parameter(torch.ones(10))
        kernel.scale = parameter

        def forward_mock(x0, x1=None):
            x0, x1 = to_tensor(x0).float(), to_tensor(x0 if x1 is None else x1).float()
            return torch.einsum("j,ij,kj->ik", parameter, x0, x1)

        kernel.forward = forward_mock
        kernel.fit(torch.rand(10, 10), np.random.rand(10))
        assert len(kernel.opt_.history) == 2
        assert all(np.isfinite(kernel.opt_.history))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_transform_of_numpy_input_keeps_the_dtype_of_the_kernel(self, kernel, dtype):
        kernel.forward = MagicMock(return_value=torch.rand(3, 4, dtype=dtype))
        kernel.x_train_ = np.random.rand(4, 5)
        transformed = kernel.transform(np.random.rand(3, 5))
        assert isinstance(transformed, np.ndarray)
        assert transformed.dtype == torch.empty(0, dtype=dtype).numpy().dtype

    def test_align_kernel_no_training_raises(self, kernel):
        kernel.x_train_ = None
        kernel.y_train_ = None
        with pytest.raises(ValueError, match="Training data must be provided"):
            kernel._align_kernel()

    def test_align_kernel_full_loop_no_early_stop(self, kernel):
        call_count = [0]

        def forward_mock(x0, x1=None):
            if x1 is None:
                x1 = x0
            call_count[0] += 1
            x0 = to_tensor(x0).float()
            x1 = to_tensor(x1).float()
            p = torch.ones_like(x0).float().requires_grad_()
            return torch.einsum("ij,ij,kj->ik", p, x0, x1)

        kernel.forward = forward_mock
        kernel.alignment_iterations = 3
        kernel.alignment_early_stopping_patience = 100
        kernel.x_train_ = torch.rand(10, 5)
        kernel.y_train_ = torch.arange(10).long()
        kernel._align_kernel()
        assert kernel.opt_ is not None

    def test_create_y_kernel_numpy_labels(self, kernel):
        kernel.x_train_ = torch.rand(10, 5)
        kernel.y_train_ = np.arange(10)
        y_kernel = kernel._create_y_kernel()
        assert isinstance(y_kernel, torch.Tensor)
        assert y_kernel.shape == (10, 10)

    def test_create_y_kernel_regression(self, kernel):
        kernel.x_train_ = torch.rand(10, 5)
        kernel.y_train_ = torch.rand(10)
        y_kernel = kernel._create_y_kernel()
        assert isinstance(y_kernel, torch.Tensor)
        assert y_kernel.shape == (10, 10)
