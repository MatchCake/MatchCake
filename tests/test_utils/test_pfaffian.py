import concurrent.futures
import threading
from functools import partial

import numpy as np
import pytest
import torch
import torch_pfaffian
from torch.autograd import gradcheck

from matchcake import utils
from matchcake.utils._pfaffian import _pfaffian_fdbpf_lock

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


class TestPfaffianFDBPfThreadSafety:
    @staticmethod
    def _skew(n):
        m = np.random.rand(n, n)
        return m - m.T

    def test_lock_exists(self):
        assert isinstance(_pfaffian_fdbpf_lock, type(threading.Lock()))

    def test_lock_held_during_call(self, monkeypatch):
        lock_state = []
        original = torch_pfaffian.get_pfaffian_function

        def capturing(name):
            fn = original(name)

            def wrapper(matrix):
                lock_state.append(_pfaffian_fdbpf_lock.locked())
                return fn(matrix)

            return wrapper

        monkeypatch.setattr(torch_pfaffian, "get_pfaffian_function", capturing)
        utils.pfaffian(self._skew(2), method="PfaffianFDBPf")
        assert lock_state == [True]

    def test_concurrent_calls_correct(self):
        n_workers = 8
        errors = []

        def run(epsilon):
            matrix = self._skew(4)
            # PfaffianFDBPf computes sqrt(|det| + epsilon), so pf^2 = |det| + epsilon
            expected = abs(np.linalg.det(matrix)) + epsilon
            pf = utils.pfaffian(matrix, method="PfaffianFDBPf", epsilon=epsilon)
            pf_sq = float(np.real(pf**2))
            if np.isnan(pf_sq):
                errors.append(f"NaN result with epsilon={epsilon}")
            elif not np.isclose(pf_sq, expected, atol=1e-6, rtol=1e-4):
                errors.append(f"epsilon={epsilon}: pf^2={pf_sq:.6g} != |det|+eps={expected:.6g}")

        epsilons = [1e-32, 1e-16, 1e-8, 1e-4] * (n_workers // 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(run, eps) for eps in epsilons]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert not errors, "\n".join(errors)
