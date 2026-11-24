import numpy as np
import pytest
import torch

from matchcake.ml.kernels.fermionic_pqc_kernel import FermionicPQCKernel
from matchcake.ml.kernels.linear_nif_kernel import LinearNIFKernel
from matchcake.ml.kernels.nystroem import Nystroem
from matchcake.utils.torch_utils import to_numpy

from ...configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON, TEST_SEED


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

    def test_transform_reproduces_gram_matrix_with_every_sample_as_component(self):
        x = np.random.default_rng(TEST_SEED).random((12, 8)).astype(np.float32)
        instance = Nystroem(kernel=FermionicPQCKernel(n_qubits=4, rotations="Y,Z"), n_components=len(x), random_state=0)
        instance.fit(x)
        with torch.no_grad():
            gram = to_numpy(instance.kernel(x))
        features = instance.transform(x)
        np.testing.assert_allclose(
            features @ features.T, gram, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_freeze(self, instance, x_train):
        instance.fit(x_train)
        instance.freeze()
        assert not instance.kernel.training
