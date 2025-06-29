from typing import Literal

import numpy as np
import pennylane as qml
import pytest
from torch.autograd import gradcheck

from matchcake import MatchgateStandardParams

from ..configs import (ATOL_APPROX_COMPARISON, ATOL_MATRIX_COMPARISON,
                       N_RANDOM_TESTS_PER_CASE, RTOL_APPROX_COMPARISON,
                       RTOL_MATRIX_COMPARISON, TEST_SEED, set_seed)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=1.0),
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
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=-1.0, x=1.0, y=1.0, z=1.0),
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
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=-1.0, y=1.0, z=-1.0),
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
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=-1.0),
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
def test_standard_params_to_matrix(
    params: MatchgateStandardParams,
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
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=1.0, y=1.0, z=1.0),
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
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=-1.0, x=1.0, y=1.0, z=1.0),
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
            MatchgateStandardParams(a=1.0, b=1.0, c=1.0, d=1.0, w=1.0, x=-1.0, y=1.0, z=-1.0),
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
    rn_tensor = torch.rand(batch_size, 4, 4, device="cpu", requires_grad=True)

    def std_sum(inputs):
        params = MatchgateStandardParams.from_matrix(inputs)
        return torch.sum(params.to_matrix())

    assert gradcheck(
        std_sum,
        (rn_tensor,),
        eps=1e-3,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params,interface",
    [
        (MatchgateStandardParams.random(), interface)
        for interface in ["numpy", "torch"]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_standard_params_to_interface(params, interface: Literal["numpy", "torch"]):
    if interface == "torch":
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed.")
    std_params = params.to_interface(interface)
    vec = std_params.to_vector()
    assert qml.math.get_interface(vec) == interface
