import numpy as np
import pytest

from matchcake.utils.math import svd
from ..configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "matrix",
    [np.random.rand(batch_size, size, size).squeeze() for size in range(2, 10) for batch_size in [1, 3]],
)
def test_orthonormalize(matrix):
    u, s, v = svd(matrix)
    pred_matrix = np.einsum("...ik,...k,...kj->...ij", u, s, v)
    np.testing.assert_allclose(pred_matrix, matrix, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)
