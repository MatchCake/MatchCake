import numpy as np
import pytest

from matchcake.operations.single_particle_transition_matrices.sptm_block_diag_angle_embedding import \
    SptmBlockDiagAngleEmbedding

@pytest.mark.parametrize(
    "batch_size, size",
    [(batch_size, size) for batch_size in [1, 3] for size in [2,3,4]],
)
class TestSingleParticleTransitionMatrixOperationBlockDiagonalAngleEmbedding:
    def test_single_particle_transition_matrix_diag_angle_embedding(self, batch_size, size):
        matrix = np.random.random((batch_size, 2 * size, 2 * size))
        sptm = SptmBlockDiagAngleEmbedding(matrix, wires=np.arange((size**2)*4))
        shape_size = (size**2)*8
        assert sptm.shape == (batch_size, shape_size, shape_size)
