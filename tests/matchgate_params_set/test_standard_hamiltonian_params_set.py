import numpy as np
import pytest

from msim import (
    MatchgateHamiltonianCoefficientsParams,
    MatchgateStandardHamiltonianParams,
)
from msim import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
)

np.random.seed(TEST_SEED)


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
        params.to_matrix(add_epsilon=False), params.epsilon
    )
    std_params = MatchgateStandardHamiltonianParams.from_matrix(hamiltonian)
    std_params_from = MatchgateStandardHamiltonianParams.parse_from_params(params)
    assert std_params == std_params_from
