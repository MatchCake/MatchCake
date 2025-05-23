import numpy as np
import pytest

from matchcake.matchgate_parameter_sets import transfer_functions
from matchcake.matchgate_parameter_sets.transfer_functions import (
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from ..configs import TEST_SEED, set_seed

MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateHamiltonianCoefficientsParams.ALLOW_COMPLEX_PARAMS = (
    True  # TODO: remove this line
)
MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "cls_list,target_cls,expected_cls",
    [
        (
            [MatchgatePolarParams, MatchgateStandardParams],
            MatchgateStandardHamiltonianParams,
            MatchgateStandardParams,
        )
    ],
)
def test_get_closest_cls(cls_list, target_cls, expected_cls):
    predicted_cls = transfer_functions.get_closest_cls(cls_list, target_cls)
    assert predicted_cls == expected_cls
