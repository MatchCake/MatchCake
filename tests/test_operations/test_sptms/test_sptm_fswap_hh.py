import numpy as np

from matchcake.operations.single_particle_transition_matrices.sptm_fswap_hh import SptmFSwapHH

from ...configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED, set_seed


class TestSptmFSwapHH:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_random_returns_instance(self):
        op = SptmFSwapHH.random(wires=[0, 1])
        assert op is not None

    def test_init_matrix_shape(self):
        op = SptmFSwapHH(wires=[0, 1])
        mat = op.matrix()
        assert mat.shape == (4, 4)

    def test_init_four_wires(self):
        op = SptmFSwapHH(wires=[0, 1, 2, 3])
        mat = op.matrix()
        assert mat.shape == (8, 8)

    def test_adjoint_returns_self(self):
        op = SptmFSwapHH(wires=[0, 1])
        adj = op.adjoint()
        np.testing.assert_allclose(
            adj.matrix(),
            op.matrix(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
