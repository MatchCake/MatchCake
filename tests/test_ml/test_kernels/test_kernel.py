from unittest.mock import MagicMock

import pytest
import torch

from matchcake.ml.kernels.kernel import Kernel


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
