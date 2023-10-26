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


@pytest.mark.parametrize(
    "state,hamming_weight",
    [
        #          0
        (np.array([1, 0]), 0),
        (np.array([0, 1]), 1),
        #          0     1
        (np.array([1, 0, 1, 0]), 0),
        (np.array([1, 0, 0, 1]), 1),
        (np.array([0, 1, 0, 1]), 2),
        #          0     1     2
        (np.array([1, 0, 1, 0, 1, 0]), 0),
        (np.array([1, 0, 1, 0, 0, 1]), 1),
        (np.array([1, 0, 0, 1, 0, 1]), 2),
        (np.array([0, 1, 0, 1, 0, 1]), 3),
        #          0     1     2     3
        (np.array([1, 0, 1, 0, 1, 0, 1, 0]), 0),
        (np.array([1, 0, 1, 0, 1, 0, 0, 1]), 1),
        (np.array([1, 0, 1, 0, 0, 1, 0, 1]), 2),
        (np.array([1, 0, 0, 1, 0, 1, 0, 1]), 3),
        (np.array([0, 1, 0, 1, 0, 1, 0, 1]), 4),
    ]
)
def test_get_hamming_weight(state, hamming_weight):
    out_hamming_weight = utils.get_hamming_weight(state)
    assert out_hamming_weight == hamming_weight, (f"The output hamming weight is not the correct one. "
                                                  f"Got {out_hamming_weight} instead of {hamming_weight}")

