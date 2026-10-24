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
    signed_pfaffian_complex,
)

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)

# A Pfaffian whose imaginary part survives is O(1); this only has to exclude an exact zero,
# which is what the pre-0.0.5 truncating kernel would have returned.
MIN_NONZERO_IMAGINARY_PART = 1e-3


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

    @staticmethod
    def pfaffian_by_expansion(matrix):
        # Independent oracle: the recursive first-row expansion
        # Pf(A) = sum_{j>0} (-1)^(j+1) A[0, j] Pf(A with row/col 0 and j removed).
        size = matrix.shape[0]
        if size == 0:
            return np.array(1.0 + 0.0j)
        total = 0.0 + 0.0j
        for column in range(1, size):
            keep = [index for index in range(1, size) if index != column]
            minor = matrix[np.ix_(keep, keep)]
            total += ((-1) ** (column + 1)) * matrix[0, column] * TestPfaffian.pfaffian_by_expansion(minor)
        return total

    @staticmethod
    def complex_skew_symmetric(n, batch_shape=(), dtype=torch.complex128, seed=0):
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(*batch_shape, n, n, dtype=dtype, generator=generator)
        return matrix - matrix.transpose(-1, -2)

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
    @pytest.mark.parametrize("batch_shape", [(7,), (3, 5), (2, 3, 4)])
    # chunk sizes 2, 3 and 4 leave a non-empty remainder chunk for the (7,) and (3, 5) batches
    # (numel 7 and 15), exercising the case where the batch is not divisible by chunk_size.
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 100])
    def test_pfaffian_chunked_matches_unchunked(self, sign, batch_shape, chunk_size):
        matrix = self.skew_symmetric(6, batch_size=None)
        matrix = np.broadcast_to(matrix, batch_shape + matrix.shape).copy()
        for index in np.ndindex(*batch_shape):
            matrix[index] = self.skew_symmetric(6, batch_size=None)
        unchunked = np.asarray(pfaffian(matrix, sign=sign))
        chunked = np.asarray(pfaffian(matrix, sign=sign, chunk_size=chunk_size))
        assert chunked.shape == unchunked.shape == batch_shape
        np.testing.assert_allclose(chunked, unchunked, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_chunk_size_single_matrix_is_noop(self, sign):
        matrix = self.skew_symmetric(4)
        chunked = pfaffian(matrix, sign=sign, chunk_size=1)
        unchunked = pfaffian(matrix, sign=sign)
        np.testing.assert_allclose(float(chunked), float(unchunked), atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("chunk_size", [1, 3, 100])
    def test_signed_pfaffian_chunk_size_passthrough(self, chunk_size):
        matrix = self.skew_symmetric(6, batch_size=7)
        unchunked = np.asarray(signed_pfaffian(matrix))
        chunked = np.asarray(signed_pfaffian(matrix, chunk_size=chunk_size))
        np.testing.assert_allclose(chunked, unchunked, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("chunk_size", [1, 3, 100])
    def test_sector_pfaffian_chunk_size_passthrough(self, chunk_size):
        cov_matrix = self.skew_symmetric(8, batch_size=7)
        index_sets = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]])
        unchunked = np.asarray(sector_pfaffian_features(cov_matrix, index_sets))
        chunked = np.asarray(sector_pfaffian_features(cov_matrix, index_sets, chunk_size=chunk_size))
        np.testing.assert_allclose(chunked, unchunked, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_pfaffian_chunked_grads_on_skew_manifold(self):
        n = 6
        n_upper = n * (n - 1) // 2
        theta = torch.randn(5, n_upper, dtype=torch.float64).requires_grad_()

        def chunked_signed(flat_upper):
            upper = torch.zeros(flat_upper.shape[0], n, n, dtype=flat_upper.dtype)
            idx = torch.triu_indices(n, n, offset=1)
            upper = upper.clone()
            upper[:, idx[0], idx[1]] = flat_upper
            matrix = upper - upper.transpose(-1, -2)
            return pfaffian(matrix, sign=True, chunk_size=2)

        assert gradcheck(chunked_signed, (theta,), atol=ATOL_APPROX_COMPARISON, rtol=10 * RTOL_APPROX_COMPARISON)

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

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_signed_pfaffian_complex_matches_the_recursive_expansion(self, n):
        matrix = self.complex_skew_symmetric(n, seed=n)
        np.testing.assert_allclose(
            complex(signed_pfaffian_complex(matrix)),
            complex(self.pfaffian_by_expansion(matrix.numpy())),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_signed_pfaffian_complex_squared_is_the_determinant(self, n):
        matrix = self.complex_skew_symmetric(n, seed=n + 20)
        np.testing.assert_allclose(
            complex(signed_pfaffian_complex(matrix) ** 2),
            complex(torch.linalg.det(matrix)),
            atol=10 * ATOL_SCALAR_COMPARISON,
            rtol=10 * RTOL_SCALAR_COMPARISON,
        )

    def test_signed_pfaffian_complex_preserves_the_imaginary_part(self):
        # Regression guard for the TorchPfaffian < 0.0.5 behaviour, where the signed path silently
        # returned the Pfaffian of the real part. A matrix whose real part has Pfaffian zero makes
        # that failure unmissable: the truncating implementation returns 0, the correct one does not.
        matrix = torch.tensor(
            [[0.0, 2.0j, 0.0, 0.0], [-2.0j, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0j], [0.0, 0.0, -3.0j, 0.0]],
            dtype=torch.complex128,
        )
        result = complex(signed_pfaffian_complex(matrix))
        np.testing.assert_allclose(result, -6.0 + 0.0j, atol=ATOL_SCALAR_COMPARISON)
        assert (
            abs(complex(signed_pfaffian_complex(self.complex_skew_symmetric(4, seed=5))).imag)
            > MIN_NONZERO_IMAGINARY_PART
        )

    def test_pfaffian_signed_preserves_the_imaginary_part(self):
        # The same guard one level down, on the function signed_pfaffian_complex delegates to.
        matrix = self.complex_skew_symmetric(6, seed=77)
        np.testing.assert_allclose(
            complex(pfaffian(matrix, sign=True)),
            complex(self.pfaffian_by_expansion(matrix.numpy())),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("batch_shape", [(3,), (2, 4)])
    def test_signed_pfaffian_complex_supports_leading_batch_dimensions(self, batch_shape):
        matrix = self.complex_skew_symmetric(4, batch_shape=batch_shape, seed=13)
        batched = signed_pfaffian_complex(matrix)
        assert tuple(batched.shape) == batch_shape
        for index in np.ndindex(*batch_shape):
            np.testing.assert_allclose(
                complex(batched[index]),
                complex(signed_pfaffian_complex(matrix[index])),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )

    def test_signed_pfaffian_complex_keeps_the_input_backend(self):
        matrix = self.complex_skew_symmetric(4, seed=31)
        assert isinstance(signed_pfaffian_complex(matrix), torch.Tensor)
        assert isinstance(signed_pfaffian_complex(matrix.numpy()), np.ndarray)

    @pytest.mark.parametrize(
        "in_dtype, expected_dtype",
        [(torch.complex64, torch.complex64), (torch.complex128, torch.complex128)],
    )
    def test_signed_pfaffian_complex_infers_the_complex_working_precision(self, in_dtype, expected_dtype):
        matrix = self.complex_skew_symmetric(4, dtype=in_dtype, seed=41)
        assert signed_pfaffian_complex(matrix).dtype == expected_dtype

    def test_signed_pfaffian_complex_explicit_dtype_override(self):
        matrix = self.complex_skew_symmetric(4, dtype=torch.complex64, seed=43)
        result = signed_pfaffian_complex(matrix, dtype=torch.complex128)
        assert result.dtype == torch.complex128
        np.testing.assert_allclose(
            complex(result),
            complex(signed_pfaffian_complex(matrix)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_signed_pfaffian_complex_on_a_real_input_returns_a_complex_result(self):
        # A real input must not be cast back down: the whole point of this function is that the
        # imaginary part survives, so the result dtype follows the working precision, not the input.
        matrix = torch.from_numpy(self.skew_symmetric(4))
        result = signed_pfaffian_complex(matrix)
        assert result.dtype == torch.complex128
        np.testing.assert_allclose(
            complex(result).real,
            float(signed_pfaffian(matrix)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_signed_pfaffian_complex_odd_size_is_zero(self):
        matrix = self.complex_skew_symmetric(3, seed=51)
        np.testing.assert_allclose(complex(signed_pfaffian_complex(matrix)), 0.0 + 0.0j, atol=ATOL_SCALAR_COMPARISON)

    def test_signed_pfaffian_complex_via_utils_namespace(self):
        matrix = torch.tensor([[0.0, 3.0 + 1.0j], [-3.0 - 1.0j, 0.0]], dtype=torch.complex128)
        np.testing.assert_allclose(
            complex(utils.signed_pfaffian_complex(matrix)), 3.0 + 1.0j, atol=ATOL_SCALAR_COMPARISON
        )

    def test_signed_pfaffian_complex_grads_on_skew_manifold(self):
        # As for the real signed path, perturbations must preserve skew-symmetry; here the
        # parameterization is complex, so real and imaginary upper triangles vary independently.
        n = 4
        n_upper = n * (n - 1) // 2
        theta = torch.randn(2, n_upper, dtype=torch.float64).requires_grad_()

        def complex_skew_from_upper(flat_upper):
            entries = flat_upper[0] + 1j * flat_upper[1]
            upper = torch.zeros(n, n, dtype=torch.complex128)
            idx = torch.triu_indices(n, n, offset=1)
            upper = upper.clone()
            upper[idx[0], idx[1]] = entries
            return signed_pfaffian_complex(upper - upper.transpose(-1, -2))

        assert gradcheck(
            complex_skew_from_upper, (theta,), atol=ATOL_APPROX_COMPARISON, rtol=10 * RTOL_APPROX_COMPARISON
        )

    def test_signed_pfaffian_complex_rejects_a_real_dtype_override(self):
        """A real working dtype truncates the input before the reduction, so the result would be
        Pf(Re M), which is not Re(Pf M). Silently upgrading the request to complex would ignore an
        explicit argument; returning the degraded value labelled complex (what an earlier draft of
        this port did) is the exact failure mode the torchpfaffian floor bump exists to remove.
        """
        matrix = self.complex_skew_symmetric(4, seed=97)
        assert matrix.imag.abs().max() > 0
        with pytest.raises(ValueError, match="requires a complex working dtype"):
            signed_pfaffian_complex(matrix, dtype=torch.float64)
        with pytest.raises(ValueError, match="requires a complex working dtype"):
            signed_pfaffian_complex(matrix, dtype=torch.float32)
