import numpy as np
import pytest

from matchcake.operations.single_particle_transition_matrices.sptm_angle_embedding import SptmAngleEmbedding


class TestSptmAngleEmbedding:
    def test_init_even_features(self):
        features = np.array([0.1, 0.2, 0.3, 0.4])
        op = SptmAngleEmbedding(features, wires=[0, 1, 2, 3])
        assert op is not None

    def test_pad_params_odd_features(self):
        features = np.array([0.1, 0.2, 0.3])
        padded = SptmAngleEmbedding.pad_params(features)
        assert padded.shape[-1] % 2 == 0
        assert padded.shape[-1] == 4

    def test_init_odd_features_gets_padded(self):
        features = np.array([0.1, 0.2, 0.3])
        op = SptmAngleEmbedding(features, wires=[0, 1, 2, 3])
        assert op is not None

    def test_too_many_features_raises(self):
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        with pytest.raises(ValueError):
            SptmAngleEmbedding(features, wires=[0, 1])

    def test_ndim_params(self):
        features = np.array([0.1, 0.2])
        op = SptmAngleEmbedding(features, wires=[0, 1])
        assert op.ndim_params == (1,)

    def test_repr(self):
        features = np.array([0.1, 0.2])
        op = SptmAngleEmbedding(features, wires=[0, 1])
        r = repr(op)
        assert "SptmAngleEmbedding" in r
