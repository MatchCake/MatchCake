import pytest
import numpy as np
from msim import utils


@pytest.mark.parametrize(
    "input_vector,target_matrix",
    [
        (
            np.array([1]),
            np.array([
                [0, 1],
                [-1, 0],
            ])
        ),
        (
            np.array([1, 2]),
            ValueError
        ),
        (
            np.array([1, 2, 3]),
            np.array([
                [0, 1, 2],
                [-1, 0, 3],
                [-2, -3, 0]
            ])
        ),
        (
            np.array([1, 2, 3, 4]),
            ValueError
        ),
        (
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([
                [0, 1, 2, 3],
                [-1, 0, 4, 5],
                [-2, -4, 0, 6],
                [-3, -5, -6, 0]
            ])
        ),
    ]
)
def test_skew_antisymmetric_vector_to_matrix(input_vector, target_matrix):
    if isinstance(target_matrix, np.ndarray):
        out_matrix = utils.skew_antisymmetric_vector_to_matrix(input_vector)
        assert np.allclose(out_matrix, target_matrix), (f"The output matrix is not the correct one. "
                                                        f"Got {out_matrix} instead of {target_matrix}")

    elif issubclass(target_matrix, BaseException):
        with pytest.raises(target_matrix):
            out_matrix = utils.skew_antisymmetric_vector_to_matrix(input_vector)


