import pytest
import numpy as np
from msim.matchgate_parameter_sets import transfer_functions
from msim.matchgate_parameter_sets.transfer_functions import (
    _transfer_funcs_by_type,
    _NODE_ORDER,
    infer_transfer_func,
    all_pairs_dijkstra_commutative_paths,
    params_to,
    MatchgateParams,
    MatchgatePolarParams,
    MatchgateStandardParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from msim.utils import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_I,
)
from ..configs import N_RANDOM_TESTS_PER_CASE
MatchgatePolarParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateHamiltonianCoefficientsParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line
MatchgateComposedHamiltonianParams.ALLOW_COMPLEX_PARAMS = True  # TODO: remove this line

np.random.seed(42)


@pytest.mark.parametrize(
    "cls_list,target_cls,expected_cls",
    [
        (
            [MatchgatePolarParams, MatchgateStandardParams],
            MatchgateStandardHamiltonianParams,
            MatchgateStandardParams
        )
    ]
)
def test_get_closest_cls(cls_list, target_cls, expected_cls):
    predicted_cls = transfer_functions.get_closest_cls(cls_list, target_cls)
    assert predicted_cls == expected_cls




