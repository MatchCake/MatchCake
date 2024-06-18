import pytest

from matchcake import mps


def test_matchgate_params_constructor_with_torch_wo_cuda():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, mps.MatchgatePolarParams.N_PARAMS, device="cpu")
    params = mps.MatchgatePolarParams(rn_tensor)
    assert isinstance(params.to_tensor(), torch.Tensor)
    assert params.to_tensor().is_cpu
    assert not params.is_cuda
    assert params.to_tensor().shape == (batch_size, mps.MatchgatePolarParams.N_PARAMS)


def test_matchgate_params_constructor_with_torch_cuda():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    batch_size = 2
    rn_tensor = torch.rand(batch_size, mps.MatchgatePolarParams.N_PARAMS, device="cuda")
    params = mps.MatchgatePolarParams(rn_tensor)
    assert isinstance(params.to_tensor(), torch.Tensor)
    assert params.to_tensor().is_cuda
    assert params.is_cuda
    assert params.to_tensor().shape == (batch_size, mps.MatchgatePolarParams.N_PARAMS)





