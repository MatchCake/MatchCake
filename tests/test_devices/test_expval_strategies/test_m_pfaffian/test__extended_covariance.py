import numpy as np
import pytest
import torch

from matchcake.devices.expval_strategies.m_pfaffian import displacement_vector
from tests.configs import ATOL_MATRIX_COMPARISON


def _random_product_state(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    state = np.zeros((n, 2), dtype=complex)
    for k in range(n):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        state[k, 0] = np.cos(theta / 2)
        state[k, 1] = np.exp(1j * phi) * np.sin(theta / 2)
    return state


class TestDisplacementVector:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_single_output_shape(self, n: int):
        state = torch.as_tensor(_random_product_state(n, seed=0), dtype=torch.complex128)
        d = displacement_vector(state, list(range(n)))
        assert d.shape == (2 * n,), f"Expected ({2 * n},), got {d.shape}"

    @pytest.mark.parametrize("batch_size", [1, 3, 5])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_batched_output_shape(self, batch_size: int, n: int):
        batch = torch.as_tensor(
            np.stack([_random_product_state(n, seed=s) for s in range(batch_size)]),
            dtype=torch.complex128,
        )
        d = displacement_vector(batch, list(range(n)))
        assert d.shape == (batch_size, 2 * n), f"Expected ({batch_size}, {2 * n}), got {d.shape}"

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batched_matches_single(self, n: int, batch_size: int):
        states_np = [_random_product_state(n, seed=s) for s in range(batch_size)]
        batch = torch.as_tensor(np.stack(states_np), dtype=torch.complex128)
        d_batch = displacement_vector(batch, list(range(n)))
        for i, state_np in enumerate(states_np):
            d_single = displacement_vector(torch.as_tensor(state_np, dtype=torch.complex128), list(range(n)))
            np.testing.assert_allclose(d_batch[i].numpy(), d_single.numpy(), atol=ATOL_MATRIX_COMPARISON)
