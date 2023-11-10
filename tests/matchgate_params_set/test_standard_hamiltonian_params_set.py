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
        MatchgateHamiltonianCoefficientsParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_parse_from_hamiltonian_coeffs_with_slow_method(
        params: MatchgateHamiltonianCoefficientsParams,
):
    hamiltonian = utils.get_non_interacting_fermionic_hamiltonian_from_coeffs(
        params.to_matrix(), params.epsilon
    )
    elements_indexes_as_array = np.asarray(MatchgateStandardHamiltonianParams.ELEMENTS_INDEXES)
    params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    std_params = MatchgateStandardHamiltonianParams.from_numpy(params_arr)
    std_params_from = MatchgateStandardHamiltonianParams.parse_from_params(params)
    assert std_params == std_params_from





