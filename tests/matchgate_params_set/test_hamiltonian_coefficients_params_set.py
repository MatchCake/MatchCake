import numpy as np
import pytest

from msim import (
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
)
from ..configs import N_RANDOM_TESTS_PER_CASE, TEST_SEED

np.random.seed(TEST_SEED)


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams(h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0),
        MatchgateHamiltonianCoefficientsParams(h0=1.0, h1=1.0, h2=1.0, h3=1.0, h4=1.0, h5=1.0),
        MatchgateHamiltonianCoefficientsParams(h0=-1.0, h1=-1.0, h2=-1.0, h3=1.0, h4=1.0, h5=1.0),
    
    ]
)
def test_parse_to_composed_hamiltonian_back_and_forth(
        params: MatchgateHamiltonianCoefficientsParams,
):
    composed_h_params = MatchgateComposedHamiltonianParams.parse_from_params(params)
    params_ = MatchgateHamiltonianCoefficientsParams.parse_from_params(composed_h_params)
    assert params_ == params


@pytest.mark.parametrize(
    "vector",
    [
        np.random.rand(MatchgateHamiltonianCoefficientsParams.N_PARAMS)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_from_numpy(
        vector: np.ndarray,
):
    params = MatchgateHamiltonianCoefficientsParams.from_numpy(vector)
    assert np.allclose(params.to_numpy(), vector)
    
    params_ = MatchgateHamiltonianCoefficientsParams.parse_from_params(params)
    assert params_ == params
    
    params__ = MatchgateHamiltonianCoefficientsParams(
        h0=vector[0],
        h1=vector[1],
        h2=vector[2],
        h3=vector[3],
        h4=vector[4],
        h5=vector[5],
        epsilon=vector[6],
        backend="numpy",
    )
    assert params__ == params


@pytest.mark.parametrize(
    "params",
    [
        MatchgateHamiltonianCoefficientsParams.random()
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_compute_hamiltonian(params):
    hamiltonian = params.compute_hamiltonian()
    elements_indexes_as_array = np.asarray(MatchgateStandardHamiltonianParams.ELEMENTS_INDEXES)
    params_arr = hamiltonian[elements_indexes_as_array[:, 0], elements_indexes_as_array[:, 1]]
    std_params = MatchgateStandardHamiltonianParams.from_numpy(params_arr)
    std_params_from = MatchgateStandardHamiltonianParams.parse_from_params(params)
    assert std_params == std_params_from
