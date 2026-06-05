import concurrent.futures
import threading
from functools import partial

import numpy as np
import pytest
import torch
import torch_pfaffian
from torch.autograd import gradcheck

from matchcake import utils
from matchcake.utils._pfaffian import (
    _pfaffian_fdbpf_lock,
    infer_real_dtype,
    sector_pfaffian_features,
    signed_pfaffian,
)

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

    def test_signed_pfaffian_numpy_input(self):
        m = np.array([[0.0, 3.0], [-3.0, 0.0]])
        pf = float(signed_pfaffian(m))
        np.testing.assert_allclose(pf, 3.0, atol=1e-12)

    def test_signed_pfaffian_empty_matrix(self):
        m = torch.zeros(0, 0, dtype=torch.float64)
        result = signed_pfaffian(m)
        np.testing.assert_allclose(float(result), 1.0, atol=1e-12)

    def test_signed_pfaffian_odd_size(self):
        m = torch.zeros(3, 3, dtype=torch.float64)
        result = signed_pfaffian(m)
        np.testing.assert_allclose(float(result), 0.0, atol=1e-12)

    def test_signed_pfaffian_near_singular_pivot(self):
        # A matrix whose first pivot is exactly zero forces sign=0.0 path.
        m = torch.zeros(4, 4, dtype=torch.float64)
        pf = float(signed_pfaffian(m))
        np.testing.assert_allclose(pf, 0.0, atol=1e-12)

    def test_sector_pfaffian_2x2_fast_path(self):
        cov = torch.tensor(
            [[0.0, 2.5, 0.0, 0.0], [-2.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.3], [0.0, 0.0, -1.3, 0.0]],
            dtype=torch.float64,
        )
        index_sets = np.array([[0, 1], [2, 3]])
        result = sector_pfaffian_features(cov, index_sets)
        np.testing.assert_allclose(result.numpy(), [2.5, 1.3], atol=1e-12)

    def test_sector_pfaffian_4x4_submatrix(self):
        a, b, c, d, e, f = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        A = torch.tensor(
            [
                [0, a, b, c],
                [-a, 0, d, e],
                [-b, -d, 0, f],
                [-c, -e, -f, 0],
            ],
            dtype=torch.float64,
        )
        expected = a * f - b * e + c * d
        index_sets = np.array([[0, 1, 2, 3]])
        result = sector_pfaffian_features(A, index_sets)
        np.testing.assert_allclose(float(result[0]), expected, atol=1e-10)

    def test_sector_pfaffian_2x2_grads(self):
        def fn(cov):
            return sector_pfaffian_features(cov, np.array([[0, 1], [2, 3]]))

        cov = torch.tensor(
            [[0.0, 2.5, 0.0, 0.0], [-2.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.3], [0.0, 0.0, -1.3, 0.0]],
            dtype=torch.float64,
        ).requires_grad_(True)
        from torch.autograd import gradcheck

        assert gradcheck(fn, (cov,), atol=1e-6)

    def test_sector_pfaffian_4x4_grads(self):
        def fn(A):
            return sector_pfaffian_features(A, np.array([[0, 1, 2, 3]]))

        m = torch.randn(4, 4, dtype=torch.float64)
        A = (m - m.T).requires_grad_(True)
        from torch.autograd import gradcheck

        assert gradcheck(fn, (A,), atol=1e-6)

    def test_signed_pfaffian_4x4_non_zero_schur_complement(self):
        a, b, c, d, e, f = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        A = torch.tensor(
            [
                [0, a, b, c],
                [-a, 0, d, e],
                [-b, -d, 0, f],
                [-c, -e, -f, 0],
            ],
            dtype=torch.float64,
        )
        expected = a * f - b * e + c * d
        pf = float(signed_pfaffian(A))
        np.testing.assert_allclose(pf, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "in_dtype, expected_real_dtype",
        [
            (torch.float32, torch.float32),
            (torch.float64, torch.float64),
            (torch.complex64, torch.float32),
            (torch.complex128, torch.float64),
        ],
    )
    def test_infer_real_dtype_torch(self, in_dtype, expected_real_dtype):
        assert infer_real_dtype(torch.zeros(2, 2, dtype=in_dtype)) == expected_real_dtype

    def test_infer_real_dtype_integer_fallback(self):
        assert infer_real_dtype(torch.zeros(2, 2, dtype=torch.int64)) == torch.float64

    def test_infer_real_dtype_numpy(self):
        assert infer_real_dtype(np.zeros((2, 2), dtype=np.float32)) == torch.float32
        assert infer_real_dtype(np.zeros((2, 2), dtype=np.complex128)) == torch.float64

    @pytest.mark.parametrize("submatrix_size", [2, 4])
    def test_sector_pfaffian_preserves_input_precision(self, submatrix_size):
        # float32 input must not be silently upcast to float64 internals.
        m = torch.randn(submatrix_size, submatrix_size, dtype=torch.float32)
        A = m - m.T
        index_sets = np.array([list(range(submatrix_size))])
        result = sector_pfaffian_features(A, index_sets)
        assert result.dtype == torch.float32

    @pytest.mark.parametrize("submatrix_size", [2, 4])
    def test_sector_pfaffian_explicit_dtype_override(self, submatrix_size):
        m = torch.randn(submatrix_size, submatrix_size, dtype=torch.float32)
        A = m - m.T
        index_sets = np.array([list(range(submatrix_size))])
        result = sector_pfaffian_features(A, index_sets, dtype=torch.float64)
        assert result.dtype == torch.float32  # output recast to input dtype
        # Value matches the float64 reference within float32 tolerance.
        ref = sector_pfaffian_features(A.to(torch.float64), index_sets)
        np.testing.assert_allclose(result.numpy(), ref.numpy(), atol=1e-5, rtol=1e-4)

    def test_signed_pfaffian_preserves_float32(self):
        m = torch.randn(4, 4, dtype=torch.float32)
        A = m - m.T
        assert signed_pfaffian(A).dtype == torch.float32

    def test_signed_pfaffian_pivot_swap_triggered(self):
        # Construct a matrix where M[3,0] > M[1,0] to force pivot swap.
        A = torch.tensor(
            [
                [0, 0.01, 0.01, 5.0],
                [-0.01, 0, 1.0, 1.0],
                [-0.01, -1.0, 0, 1.0],
                [-5.0, -1.0, -1.0, 0],
            ],
            dtype=torch.float64,
        )
        det = float(torch.linalg.det(A))
        pf = float(signed_pfaffian(A))
        np.testing.assert_allclose(pf**2, det, atol=1e-8)


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
