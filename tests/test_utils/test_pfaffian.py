from functools import partial

import numpy as np
import pytest

from matchcake import utils

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)


class TestPfaffian:
    @staticmethod
    def gen_skew_symmetric_matrix_and_det(n, batch_size=None):
        if batch_size is None:
            matrix = np.random.rand(n, n)
        else:
            matrix = np.random.rand(batch_size, n, n)
        matrix = matrix - np.einsum("...ij->...ji", matrix)
        return matrix, np.linalg.det(matrix)

    @pytest.mark.parametrize(
        "n, batch_size, mth",
        [
            (i, batch_size, mth)
            for i in np.linspace(2, 6, endpoint=True, dtype=int)
            for mth in ["det", "PfaffianFDBPf"]
            for batch_size in [None, 3]
        ],
    )
    def test_pfaffian_methods(self, n, batch_size, mth):
        matrix, target_det = self.gen_skew_symmetric_matrix_and_det(n, batch_size=batch_size)
        pf = utils.pfaffian(matrix, method=mth)
        np.testing.assert_allclose(
            pf**2, target_det, atol=10 * ATOL_SCALAR_COMPARISON, rtol=10 * RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize(
        "n, batch_size, mth",
        [
            (i, batch_size, mth)
            for i in np.linspace(2, 6, endpoint=True, dtype=int)
            for mth in ["det", "PfaffianFDBPf"]
            for batch_size in [None, 3]
        ],
    )
    def test_pfaffian_methods_grads(self, n, batch_size, mth):
        import torch
        from torch.autograd import gradcheck

        if batch_size is None:
            np_matrix = np.random.rand(n, n)
        else:
            np_matrix = np.random.rand(batch_size, n, n)
        np_matrix = np_matrix - np.einsum("...ij->...ji", np_matrix)
        torch_matrix = torch.from_numpy(np_matrix).requires_grad_()
        func = partial(utils.pfaffian, method=mth)
        assert gradcheck(
            func,
            (torch_matrix,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=10 * RTOL_APPROX_COMPARISON,
        )
