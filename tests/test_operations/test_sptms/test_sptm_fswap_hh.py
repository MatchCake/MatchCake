import numpy as np
import pytest

from matchcake.operations.single_particle_transition_matrices.sptm_fswap_hh import SptmFSwapHH
from matchcake.utils.torch_utils import to_numpy

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
        mat = op.sptm_matrix()
        assert mat.shape == (4, 4)

    def test_init_four_wires(self):
        op = SptmFSwapHH(wires=[0, 1, 2, 3])
        mat = op.sptm_matrix()
        assert mat.shape == (8, 8)

    @pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2, 3]])
    def test_adjoint_is_transpose(self, wires):
        op = SptmFSwapHH(wires=wires)
        np.testing.assert_allclose(
            to_numpy(op.adjoint().sptm_matrix()),
            to_numpy(op.sptm_matrix()).T,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2, 3]])
    def test_adjoint_inverts_matrix(self, wires):
        op = SptmFSwapHH(wires=wires)
        product = to_numpy(op.sptm_matrix()) @ to_numpy(op.adjoint().sptm_matrix())
        np.testing.assert_allclose(
            product,
            np.eye(product.shape[-1]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
