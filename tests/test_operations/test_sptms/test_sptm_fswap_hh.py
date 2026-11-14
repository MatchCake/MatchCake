import numpy as np
import pytest

import matchcake as mc
from matchcake.operations import CompHH, fSWAP
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

    @pytest.mark.parametrize("n_wires", range(2, 8))
    def test_matrix_is_orthogonal(self, n_wires):
        matrix = to_numpy(SptmFSwapHH(wires=list(range(n_wires))).sptm_matrix())

        np.testing.assert_allclose(
            matrix @ matrix.T,
            np.eye(2 * n_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("n_wires", range(2, 8))
    def test_matches_fswap_then_comp_hh_on_each_pair(self, n_wires):
        ops = [
            gate
            for wire in range(0, n_wires - 1, 2)
            for gate in (fSWAP(wires=[wire, wire + 1]), CompHH(wires=[wire, wire + 1]))
        ]
        device = mc.NIFDevice(wires=n_wires)
        device.execute_generator(iter(ops), reset=True, apply=True, cache_global_sptm=True)

        np.testing.assert_allclose(
            to_numpy(SptmFSwapHH(wires=list(range(n_wires))).sptm_matrix()),
            device.apply_metadata["global_sptm"],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_wires", [3, 5, 7])
    def test_odd_number_of_wires_leaves_last_wire_unchanged(self, n_wires):
        matrix = to_numpy(SptmFSwapHH(wires=list(range(n_wires))).sptm_matrix())

        np.testing.assert_allclose(matrix[-2:, -2:], np.eye(2), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(matrix[-2:, :-2], 0.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(matrix[:-2, -2:], 0.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
    def test_adjoint_is_transpose(self, wires):
        op = SptmFSwapHH(wires=wires)
        np.testing.assert_allclose(
            to_numpy(op.adjoint().sptm_matrix()),
            to_numpy(op.sptm_matrix()).T,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
    def test_adjoint_inverts_matrix(self, wires):
        op = SptmFSwapHH(wires=wires)
        product = to_numpy(op.sptm_matrix()) @ to_numpy(op.adjoint().sptm_matrix())
        np.testing.assert_allclose(
            product,
            np.eye(product.shape[-1]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
