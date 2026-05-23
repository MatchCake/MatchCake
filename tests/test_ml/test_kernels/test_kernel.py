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
        y = kernel.transform(x)
        kernel.forward.assert_called_once()

    def test_fit(self, kernel):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10)
        kernel.fit(x, y)
        assert kernel.x_train_ is x
        assert kernel.y_train_ is y
        assert kernel.is_fitted_

    def test_predict(self, kernel):
        x = torch.rand(10, 10)
        y = kernel.predict(x)
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
