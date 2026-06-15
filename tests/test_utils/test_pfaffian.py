import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from matchcake import utils
from matchcake.utils._pfaffian import (
    infer_real_dtype,
    pfaffian,
    sector_pfaffian_features,
    signed_pfaffian,
)

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)


class TestPfaffian:
    @staticmethod
    def skew_symmetric(n, batch_size=None):
        if batch_size is None:
            matrix = np.random.rand(n, n)
        else:
            matrix = np.random.rand(batch_size, n, n)
        return matrix - np.einsum("...ij->...ji", matrix)

    @staticmethod
    def skew_from_upper(theta, n):
        # Build a skew-symmetric matrix from its independent upper-triangular entries so
        # that autograd perturbations stay on the skew-symmetric manifold.
        upper = torch.zeros(n, n, dtype=theta.dtype)
        idx = torch.triu_indices(n, n, offset=1)
        upper = upper.clone()
        upper[idx[0], idx[1]] = theta
        return upper - upper.transpose(-1, -2)

    @pytest.mark.parametrize("n, batch_size", [(2, None), (4, None), (2, 3), (4, 3)])
    def test_pfaffian_magnitude_squared_is_abs_det(self, n, batch_size):
        matrix = self.skew_symmetric(n, batch_size)
        pf = pfaffian(matrix, sign=False)
        np.testing.assert_allclose(
            pf**2,
            np.abs(np.linalg.det(matrix)),
            atol=10 * ATOL_SCALAR_COMPARISON,
            rtol=10 * RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("n, batch_size", [(2, None), (4, None), (2, 3), (4, 3)])
    def test_pfaffian_signed_squared_is_det(self, n, batch_size):
        matrix = self.skew_symmetric(n, batch_size)
        pf = pfaffian(matrix, sign=True)
        np.testing.assert_allclose(
            pf**2,
            np.linalg.det(matrix),
            atol=10 * ATOL_SCALAR_COMPARISON,
            rtol=10 * RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_odd_size_is_zero(self, sign):
        matrix = self.skew_symmetric(3)
        pf = pfaffian(matrix, sign=sign)
        np.testing.assert_allclose(float(pf), 0.0, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_with_zeros(self, sign):
        matrix = np.zeros((4, 4))
        pf = pfaffian(matrix, sign=sign)
        np.testing.assert_allclose(float(pf**2), 0.0, atol=10 * ATOL_SCALAR_COMPARISON)

    def test_pfaffian_preserves_numpy_backend(self):
        matrix = self.skew_symmetric(4)
        pf = pfaffian(matrix, sign=True)
        assert isinstance(pf, np.ndarray)

    def test_pfaffian_signed_via_utils_namespace(self):
        matrix = np.array([[0.0, 3.0], [-3.0, 0.0]])
        np.testing.assert_allclose(float(utils.pfaffian(matrix, sign=True)), 3.0, atol=ATOL_SCALAR_COMPARISON)

    def test_pfaffian_magnitude_grads(self):
        matrix = torch.from_numpy(self.skew_symmetric(4)).requires_grad_()
        assert gradcheck(
            lambda x: pfaffian(x, sign=False),
            (matrix,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=10 * RTOL_APPROX_COMPARISON,
        )

    def test_pfaffian_magnitude_grads_with_zeros(self):
        matrix = torch.zeros(4, 4, dtype=torch.float64).requires_grad_()
        assert gradcheck(
            lambda x: pfaffian(x, sign=False),
            (matrix,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=10 * RTOL_APPROX_COMPARISON,
        )

    def test_pfaffian_signed_grads_on_skew_manifold(self):
        # The signed Pfaffian gradient is the antisymmetric (pf/2) A^{-T}; it is only
        # consistent with finite differences when perturbations preserve skew-symmetry,
        # so gradcheck must run over the upper-triangular parameterization.
        n = 4
        theta = torch.randn(n * (n - 1) // 2, dtype=torch.float64).requires_grad_()
        assert gradcheck(
            lambda t: pfaffian(self.skew_from_upper(t, n), sign=True),
            (theta,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=10 * RTOL_APPROX_COMPARISON,
        )

    def test_signed_pfaffian_numpy_input(self):
        matrix = np.array([[0.0, 3.0], [-3.0, 0.0]])
        np.testing.assert_allclose(float(signed_pfaffian(matrix)), 3.0, atol=ATOL_SCALAR_COMPARISON)

    def test_signed_pfaffian_empty_matrix(self):
        matrix = torch.zeros(0, 0, dtype=torch.float64)
        np.testing.assert_allclose(float(signed_pfaffian(matrix)), 1.0, atol=ATOL_SCALAR_COMPARISON)

    def test_signed_pfaffian_odd_size(self):
        matrix = torch.zeros(3, 3, dtype=torch.float64)
        np.testing.assert_allclose(float(signed_pfaffian(matrix)), 0.0, atol=ATOL_SCALAR_COMPARISON)

    def test_signed_pfaffian_4x4_value(self):
        a, b, c, d, e, f = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        matrix = torch.tensor(
            [[0, a, b, c], [-a, 0, d, e], [-b, -d, 0, f], [-c, -e, -f, 0]],
            dtype=torch.float64,
        )
        np.testing.assert_allclose(float(signed_pfaffian(matrix)), a * f - b * e + c * d, atol=ATOL_SCALAR_COMPARISON)

    def test_signed_pfaffian_preserves_float32(self):
        m = torch.randn(4, 4, dtype=torch.float32)
        assert signed_pfaffian(m - m.T).dtype == torch.float32

    def test_signed_pfaffian_explicit_dtype_override(self):
        m = torch.randn(4, 4, dtype=torch.float32)
        matrix = m - m.T
        result = signed_pfaffian(matrix, dtype=torch.float64)
        assert result.dtype == torch.float32  # output recast to input dtype
        ref = signed_pfaffian(matrix.to(torch.float64))
        np.testing.assert_allclose(float(result), float(ref), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_pfaffian_explicit_dtype_override(self):
        matrix = self.skew_symmetric(4).astype(np.float32)
        result = pfaffian(matrix, sign=False, dtype=torch.float64)
        np.testing.assert_allclose(
            result**2,
            np.abs(np.linalg.det(matrix.astype(np.float64))),
            atol=10 * ATOL_SCALAR_COMPARISON,
            rtol=10 * RTOL_SCALAR_COMPARISON,
        )

    def test_signed_pfaffian_squared_is_det_with_pivot_swap(self):
        matrix = torch.tensor(
            [[0, 0.01, 0.01, 5.0], [-0.01, 0, 1.0, 1.0], [-0.01, -1.0, 0, 1.0], [-5.0, -1.0, -1.0, 0]],
            dtype=torch.float64,
        )
        np.testing.assert_allclose(
            float(signed_pfaffian(matrix)) ** 2,
            float(torch.linalg.det(matrix)),
            atol=10 * ATOL_SCALAR_COMPARISON,
        )

    def test_sector_pfaffian_2x2_fast_path(self):
        cov = torch.tensor(
            [[0.0, 2.5, 0.0, 0.0], [-2.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.3], [0.0, 0.0, -1.3, 0.0]],
            dtype=torch.float64,
        )
        result = sector_pfaffian_features(cov, np.array([[0, 1], [2, 3]]))
        np.testing.assert_allclose(result.numpy(), [2.5, 1.3], atol=ATOL_SCALAR_COMPARISON)

    def test_sector_pfaffian_4x4_submatrix(self):
        a, b, c, d, e, f = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        matrix = torch.tensor(
            [[0, a, b, c], [-a, 0, d, e], [-b, -d, 0, f], [-c, -e, -f, 0]],
            dtype=torch.float64,
        )
        result = sector_pfaffian_features(matrix, np.array([[0, 1, 2, 3]]))
        np.testing.assert_allclose(float(result[0]), a * f - b * e + c * d, atol=10 * ATOL_SCALAR_COMPARISON)

    def test_sector_pfaffian_2x2_grads(self):
        cov = torch.tensor(
            [[0.0, 2.5, 0.0, 0.0], [-2.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.3], [0.0, 0.0, -1.3, 0.0]],
            dtype=torch.float64,
        ).requires_grad_(True)
        assert gradcheck(
            lambda c: sector_pfaffian_features(c, np.array([[0, 1], [2, 3]])),
            (cov,),
            atol=ATOL_APPROX_COMPARISON,
        )

    def test_sector_pfaffian_4x4_grads_on_skew_manifold(self):
        n = 4
        theta = torch.randn(n * (n - 1) // 2, dtype=torch.float64).requires_grad_()
        assert gradcheck(
            lambda t: sector_pfaffian_features(self.skew_from_upper(t, n), np.array([[0, 1, 2, 3]])),
            (theta,),
            atol=ATOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("submatrix_size", [2, 4])
    def test_sector_pfaffian_preserves_input_precision(self, submatrix_size):
        m = torch.randn(submatrix_size, submatrix_size, dtype=torch.float32)
        result = sector_pfaffian_features(m - m.T, np.array([list(range(submatrix_size))]))
        assert result.dtype == torch.float32

    @pytest.mark.parametrize("submatrix_size", [2, 4])
    def test_sector_pfaffian_explicit_dtype_override(self, submatrix_size):
        m = torch.randn(submatrix_size, submatrix_size, dtype=torch.float32)
        matrix = m - m.T
        index_sets = np.array([list(range(submatrix_size))])
        result = sector_pfaffian_features(matrix, index_sets, dtype=torch.float64)
        assert result.dtype == torch.float32  # output recast to input dtype
        ref = sector_pfaffian_features(matrix.to(torch.float64), index_sets)
        np.testing.assert_allclose(
            result.numpy(), ref.numpy(), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

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
