from functools import partial

import numpy as np
import pytest

import pennylane as qml

from matchcake import utils
from matchcake.utils.math import orthonormalize, dagger
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_SCALAR_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


class __SpecialOrthonormalizedTensor:
    IDS = []
    def __init__(self, tensor):
        self.tensor = tensor
        self.orthonormalized_tensor = orthonormalize(tensor)
        self.ID = len(self.IDS)
        self.__class__.IDS.append(self.ID)

    def __array__(self):
        return self.orthonormalized_tensor



@pytest.mark.parametrize(
    "matrix",
    [
        np.random.uniform(1, 10) * np.random.rand(batch_size, size, size).squeeze()
        for size in range(2, 10)
        for batch_size in [1, 3]
    ]
)
def test_orthonormalize(matrix):
    ortho_matrix = orthonormalize(matrix)
    expected_eye = qml.math.einsum("...ij,...jk->...ik", ortho_matrix, dagger(ortho_matrix))
    eye = np.zeros_like(expected_eye)
    eye[..., np.arange(eye.shape[-1]), np.arange(eye.shape[-1])] = 1
    np.testing.assert_allclose(expected_eye, eye, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@pytest.mark.parametrize(
    "matrix",
    [
        np.random.uniform(1, 10) * np.random.rand(batch_size, size, size).squeeze()
        for size in range(2, 10)
        for batch_size in [1, 3]
    ]
)
def test_orthonormalize_already_orthonormalize(matrix):
    special_tensor = __SpecialOrthonormalizedTensor(matrix)
    ortho_matrix = orthonormalize(special_tensor.orthonormalized_tensor)
    np.testing.assert_allclose(special_tensor, ortho_matrix)
    assert special_tensor.orthonormalized_tensor is ortho_matrix
