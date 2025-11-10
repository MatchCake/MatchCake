from unittest.mock import MagicMock

import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.ml.kernels import NIFKernel


class TestNIFKernel:
    @pytest.fixture
    def kernel_instance(self):
        instance = NIFKernel()
        instance.ansatz = MagicMock()
        return instance

    def test_init(self, kernel_instance):
        assert kernel_instance.random_state == 0
        assert kernel_instance.gram_batch_size == 10_000
        assert isinstance(kernel_instance.q_device, NonInteractingFermionicDevice)

    def test_forward(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance(x)
        kernel_instance.ansatz.assert_called()

    def test_compute_similarities(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance.compute_similarities(x, x)
        kernel_instance.ansatz.assert_called()
