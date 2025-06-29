import numpy as np
import pytest

from matchcake import utils

from ..configs import (ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON,
                       TEST_SEED, set_seed)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "state,wires,expected_probs",
    [
        ([1, 0], [0], [1, 0]),
        ([1, 0, 0, 0], [0], [1, 0]),
        ([1, 0, 0, 0], [1], [1, 0]),
        ([1, 0, 0, 0], [0, 1], [1, 0, 0, 0]),
        ([0.5, 0, 0, np.sqrt(0.75)], [0], [0.25, 0.75]),
        ([0.5, 0, 0, np.sqrt(0.75)], [1], [0.25, 0.75]),
        ([0.5, 0, 0, np.sqrt(0.75)], [0, 1], [0.25, 0, 0, 0.75]),
    ],
)
def test_get_probabilities_from_state(state, wires, expected_probs):
    state = np.asarray(state)
    probs = utils.get_probabilities_from_state(state, wires)
    np.testing.assert_allclose(
        probs,
        expected_probs,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
