import numpy as np
import pytest

from matchcake import (
    MatchgateStandardParams,
)
from ..configs import (
    set_seed,
    TEST_SEED,
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "params,expected",
    [
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=-1.0, x=1.0, y=1.0, z=1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, -1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=-1.0, y=1.0, z=-1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, -1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
        (
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=-1.0),
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
        ),
    ],
)
def test_standard_params_to_matrix(
        params: MatchgateStandardParams,
        expected: np.ndarray,
):
    matrix = params.to_matrix()
    np.testing.assert_allclose(matrix.squeeze(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)


@pytest.mark.parametrize(
    "matrix,params",
    [
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=1.0),
        ),
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, -1, 1, 0.0],
                    [0.0, 1, 1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=-1.0, x=1.0, y=1.0, z=1.0),
        ),
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, -1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=-1.0, y=1.0, z=-1.0),
        ),
        (
                np.array([
                    [1, 0.0, 0.0, 1],
                    [0.0, 1, 1, 0.0],
                    [0.0, 1, -1, 0.0],
                    [1, 0.0, 0.0, 1],
                ]),
                MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=-1.0),
        ),
    ],
)
def test_standard_params_from_matrix(matrix, params):
    params_ = MatchgateStandardParams.from_matrix(matrix)
    assert params_ == params


def test_standard_params_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, MatchgateStandardParams.N_PARAMS, device="cpu", requires_grad=True)
    params = MatchgateStandardParams(rn_tensor)
    assert isinstance(params.to_tensor(), torch.Tensor)
    assert params.to_tensor().requires_grad
    assert params.requires_grad


def test_standard_params_from_matrix_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, 4, 4, device="cpu", requires_grad=True)
    params = MatchgateStandardParams.from_matrix(rn_tensor)
    assert isinstance(params.to_tensor(), torch.Tensor)
    assert params.to_tensor().requires_grad
    assert params.requires_grad


def test_standard_params_from_vector_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, MatchgateStandardParams.N_PARAMS, device="cpu", requires_grad=True)
    params = MatchgateStandardParams.from_vector(rn_tensor)
    assert isinstance(params.to_tensor(), torch.Tensor)
    assert params.to_tensor().requires_grad
    assert params.requires_grad


def test_standard_params_from_vector_to_matrix_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, MatchgateStandardParams.N_PARAMS, device="cpu", requires_grad=True)
    params = MatchgateStandardParams.from_vector(rn_tensor)
    matrix = params.to_matrix()
    assert isinstance(matrix, torch.Tensor)
    assert matrix.requires_grad


def test_standard_params_from_matrix_to_vector_requires_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, 4, 4, device="cpu", requires_grad=True)
    params = MatchgateStandardParams.from_matrix(rn_tensor)
    vector = params.to_vector()
    assert isinstance(vector, torch.Tensor)
    assert vector.requires_grad


def test_standard_params_from_matrix_grad_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, 4, 4, device="cpu", requires_grad=False)
    rn_matrix = MatchgateStandardParams.from_matrix(rn_tensor).to_matrix().requires_grad_(True)
    out = torch.exp(rn_matrix).sum()
    expected_gradients = torch.autograd.grad(out, rn_matrix, torch.ones_like(out))[0]

    params = MatchgateStandardParams.from_matrix(rn_tensor).requires_grad_(True)
    matrix = params.to_matrix()
    pred_out = torch.exp(matrix).sum()
    pred_out.backward()
    gradients = params.to_matrix().grad
    assert torch.allclose(gradients, expected_gradients)
