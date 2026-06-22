import numpy as np
import pytest
import torch

from matchcake.utils._pfaffian import signed_pfaffian, signed_pfaffian_complex

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)


def _recursive_pfaffian(matrix):
    """Dense reference: recursive complex-capable Pfaffian (the oracle used by verify_*.py)."""
    m = matrix.shape[0]
    if m == 0:
        return 1.0 + 0j
    if m % 2 == 1:
        return 0.0 + 0j
    if m == 2:
        return matrix[0, 1]
    total = 0.0 + 0j
    rest = list(range(1, m))
    for pos, j in enumerate(rest):
        sub = [k for k in rest if k != j]
        total += (-1) ** pos * matrix[0, j] * _recursive_pfaffian(matrix[np.ix_(sub, sub)])
    return total


class TestSignedPfaffianComplex:
    @staticmethod
    def _skew_complex(n, rng, batch_size=None):
        shape = (n, n) if batch_size is None else (batch_size, n, n)
        a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return a - np.einsum("...ij->...ji", a)

    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
    def test_matches_recursive_reference(self, n):
        rng = np.random.default_rng(n)
        matrix = self._skew_complex(n, rng)
        got = complex(signed_pfaffian_complex(matrix))
        ref = _recursive_pfaffian(matrix)
        # A genuinely complex Pfaffian (the whole point of the function).
        assert abs(ref.imag) > 1e-6
        np.testing.assert_allclose(got, ref, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_squared_is_determinant(self, n):
        rng = np.random.default_rng(100 + n)
        matrix = self._skew_complex(n, rng)
        pf = complex(signed_pfaffian_complex(matrix))
        np.testing.assert_allclose(
            pf**2, np.linalg.det(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("batch_size", [3, (2, 4)])
    def test_batched_matches_loop(self, batch_size):
        rng = np.random.default_rng(7)
        n = 4
        flat = batch_size if isinstance(batch_size, int) else int(np.prod(batch_size))
        matrix = self._skew_complex(n, rng, batch_size=flat)
        if not isinstance(batch_size, int):
            matrix = matrix.reshape(*batch_size, n, n)
        got = np.asarray(signed_pfaffian_complex(matrix))
        assert got.shape == tuple(np.shape(matrix)[:-2])
        flat_matrix = matrix.reshape(-1, n, n)
        ref = np.array([_recursive_pfaffian(flat_matrix[i]) for i in range(flat)]).reshape(got.shape)
        np.testing.assert_allclose(got, ref, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_odd_size_is_zero(self):
        rng = np.random.default_rng(1)
        matrix = self._skew_complex(3, rng)
        np.testing.assert_allclose(complex(signed_pfaffian_complex(matrix)), 0.0, atol=ATOL_SCALAR_COMPARISON)

    def test_two_by_two_value(self):
        matrix = np.array([[0, 2 + 3j], [-(2 + 3j), 0]], dtype=complex)
        np.testing.assert_allclose(complex(signed_pfaffian_complex(matrix)), 2 + 3j, atol=ATOL_SCALAR_COMPARISON)

    def test_agrees_with_real_signed_on_real_input(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(6, 6))
        matrix = a - a.T
        got = complex(signed_pfaffian_complex(matrix.astype(complex)))
        ref = float(signed_pfaffian(matrix))
        np.testing.assert_allclose(got.real, ref, atol=ATOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(got.imag, 0.0, atol=ATOL_SCALAR_COMPARISON)

    def test_torch_input_returns_torch_complex(self):
        rng = np.random.default_rng(5)
        matrix = torch.as_tensor(self._skew_complex(4, rng))
        result = signed_pfaffian_complex(matrix)
        assert isinstance(result, torch.Tensor)
        assert result.is_complex()
        np.testing.assert_allclose(complex(result), _recursive_pfaffian(matrix.numpy()), atol=ATOL_MATRIX_COMPARISON)

    def test_preserves_complex64_precision(self):
        rng = np.random.default_rng(6)
        matrix = torch.as_tensor(self._skew_complex(4, rng)).to(torch.complex64)
        assert signed_pfaffian_complex(matrix).dtype == torch.complex64

    def test_explicit_dtype_override(self):
        rng = np.random.default_rng(8)
        matrix = self._skew_complex(4, rng)
        got = complex(signed_pfaffian_complex(matrix, dtype=torch.complex128))
        np.testing.assert_allclose(got, _recursive_pfaffian(matrix), atol=ATOL_MATRIX_COMPARISON)
