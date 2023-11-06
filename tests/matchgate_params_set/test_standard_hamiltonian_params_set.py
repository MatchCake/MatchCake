import pytest
from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from msim import utils
import numpy as np
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
        for vector in np.random.rand(N_RANDOM_TESTS_PER_CASE, 6)
    ],
)
def test_parse_from_hamiltonian_coeffs_with_slow_method(
        params: MatchgateHamiltonianCoefficientsParams,
):
    std_h_params = MatchgateStandardHamiltonianParams.parse_from_params(params)

    hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(std_h_params.to_matrix())
    elements_indexes_as_array = np.asarray(MatchgateStandardParams.ELEMENTS_INDEXES)
    params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    params_ = MatchgateHamiltonianCoefficientsParams.from_numpy(params_arr)
    assert params_ == std_h_params





