import pytest
from msim import utils
import numpy as np


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
    ]
)
def test_get_probabilities_from_state(state, wires, expected_probs):
    state = np.asarray(state)
    probs = utils.get_probabilities_from_state(state, wires)
    assert np.allclose(probs, expected_probs), (
        f"The probabilities are not the correct ones. Got {probs} instead of {expected_probs}."
    )

