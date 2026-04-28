from functools import partial

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from matchcake import utils

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)


@pytest.mark.parametrize(
    "n, batch_size, mth",
    [
        (2, None, "det"),
        (3, None, "det"),
        (5, None, "det"),
        (2, 3, "det"),
        (2, 3, "PfaffianFDBPf"),
        (3, None, "PfaffianFDBPf"),
    ],
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

    def test_pfaffian_methods(self, n, batch_size, mth):
        matrix, target_det = self.gen_skew_symmetric_matrix_and_det(n, batch_size=batch_size)
        pf = utils.pfaffian(matrix, method=mth)
        np.testing.assert_allclose(
            pf**2, target_det, atol=10 * ATOL_SCALAR_COMPARISON, rtol=10 * RTOL_SCALAR_COMPARISON
        )

    def test_pfaffian_methods_grads(self, n, batch_size, mth):
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

    def test_with_zeros(self, n, batch_size, mth):
        matrix = np.zeros((n, n))
        target_det = 0.0
        pf = utils.pfaffian(matrix, method=mth)
        np.testing.assert_allclose(pf**2, target_det, atol=1e-32)

    def test_grads_with_zeros(self, n, batch_size, mth):
        if batch_size is None:
            np_matrix = np.zeros((n, n))
        else:
            np_matrix = np.zeros((batch_size, n, n))
        np_matrix = np_matrix - np.einsum("...ij->...ji", np_matrix)
        torch_matrix = torch.from_numpy(np_matrix).requires_grad_()
        func = partial(utils.pfaffian, method=mth)
        assert gradcheck(
            func,
            (torch_matrix,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=10 * RTOL_APPROX_COMPARISON,
        )


class TestPfaffianExtended:
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            utils.pfaffian(np.random.rand(2, 2), method="invalid_method")
