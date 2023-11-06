import numpy as np
import pytest

from msim import (
    MatchgateStandardParams,
    MatchgatePolarParams,
    MatchgateHamiltonianCoefficientsParams,
    MatchgateComposedHamiltonianParams,
    MatchgateStandardHamiltonianParams,
    utils,
)
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


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
        backend="numpy",
    )
    assert params__ == params

