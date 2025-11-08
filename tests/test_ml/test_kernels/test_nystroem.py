import numpy as np
import pytest

from matchcake.ml.kernels.linear_nif_kernel import LinearNIFKernel
from matchcake.ml.kernels.nystroem import Nystroem


class TestNystroem:
    @pytest.fixture
    def instance(self):
        instance = Nystroem(
            kernel=LinearNIFKernel(),
            n_components=10,
            random_state=0,
        )
        return instance

    @pytest.fixture
    def x_train(self):
        return np.linspace(0.0, 0.05, num=80).reshape(10, 8)

    def test_fit(self, instance, x_train):
        instance.fit(x_train)
        assert instance.normalization_ is not None
        assert instance.components_ is not None
        assert instance.kernel.is_fitted_

    def test_transform(self, instance, x_train):
        instance.fit(x_train)
        transformed = instance.transform(x_train)
        assert transformed.shape == (10, 10)

    def test_freeze(self, instance, x_train):
        instance.fit(x_train)
        instance.freeze()
        assert not instance.kernel.training
