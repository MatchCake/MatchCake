from unittest.mock import MagicMock

import pytest
import torch

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake.ml.kernels import NIFKernel


class TestNIFKernel:
    @pytest.fixture
    def kernel_instance(self):
        instance = NIFKernel()
        instance.ansatz_mock = MagicMock()

        def ansatz_mock_func(x):
            instance.ansatz_mock(x)
            yield MatchgateOperation.random(wires=[0, 1], batch_size=len(x))
            return

        instance.ansatz = ansatz_mock_func
        return instance

    def test_init(self, kernel_instance):
        assert kernel_instance.random_state == 0
        assert kernel_instance.gram_batch_size == 10_000
        assert isinstance(kernel_instance.q_device, NonInteractingFermionicDevice)

    def test_forward(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance(x)
        kernel_instance.ansatz_mock.assert_called()

    def test_compute_similarities(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance.compute_similarities(x, x)
        kernel_instance.ansatz_mock.assert_called()
