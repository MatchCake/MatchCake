import pytest
import numpy as np
from msim.matchgate_parameter_sets.transfer_functions import (
    _transfer_funcs_by_type,
    params_to,
    MatchgateParams,
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.mark.parametrize(
    "__from_cls,__from_params, __to_cls",
    [
        (_from_cls, _from_cls.random(), _to_cls)
        for _from_cls, _to_cls_dict in _transfer_funcs_by_type.items()
        for _to_cls in _to_cls_dict.keys()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_transfer_functions_back_and_forth(__from_cls, __from_params, __to_cls):
    to_params = _transfer_funcs_by_type[__from_cls][__to_cls](__from_params)
    _from_params = _transfer_funcs_by_type[__to_cls][__from_cls](to_params)
    assert _from_params == __from_params



