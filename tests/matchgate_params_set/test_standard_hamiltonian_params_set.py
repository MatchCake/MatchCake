import numpy as np
import pytest

from matchcake import (
    MatchgateHamiltonianCoefficientsParams,
    MatchgateStandardHamiltonianParams,
)
from matchcake import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


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


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=1, u3=1, u4=1, u5=1, u6=1, u7=1
            ),
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
        ),
        (
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=-1, u3=1, u4=1, u5=1, u6=1, u7=1
            ),
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, -1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
        ),
        (
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=1, u3=-1, u4=1, u5=-1, u6=1, u7=1
            ),
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, -1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
        ),
        (
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=1, u3=1, u4=1, u5=-1, u6=1, u7=1
            ),
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
        ),
    ],
)
def test_standard_hamiltonian_params_to_matrix(
    params: MatchgateStandardHamiltonianParams,
    expected: np.ndarray,
):
    matrix = params.to_matrix()
    np.testing.assert_allclose(
        matrix.squeeze(),
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "matrix,params",
    [
        (
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=1, u3=1, u4=1, u5=1, u6=1, u7=1
            ),
        ),
        (
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, -1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=-1, u3=1, u4=1, u5=1, u6=1, u7=1
            ),
        ),
        (
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, -1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=1, u3=-1, u4=1, u5=-1, u6=1, u7=1
            ),
        ),
        (
            np.array(
                [
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]
            ),
            MatchgateStandardHamiltonianParams(
                u0=1, u1=1, u2=1, u3=1, u4=1, u5=-1, u6=1, u7=1
            ),
        ),
    ],
)
def test_standard_hamiltonian_params_from_matrix(matrix, params):
    params_ = MatchgateStandardHamiltonianParams.from_matrix(matrix)
    assert params_ == params


def test_matchgate_gradient_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(
        batch_size,
        MatchgateStandardHamiltonianParams.N_PARAMS,
        device="cpu",
        requires_grad=True,
    )
    params = MatchgateStandardHamiltonianParams(rn_tensor)
    assert isinstance(params.to_tensor(), torch.Tensor)
    assert params.to_tensor().requires_grad
    assert params.requires_grad
