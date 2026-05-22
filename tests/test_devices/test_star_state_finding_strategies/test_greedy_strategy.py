from functools import partial

import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.star_state_finding_strategies.greedy_strategy import GreedyStrategy

from ...configs import TEST_SEED, set_seed
from .. import init_nif_device


def _setup_zero_state_device(n_wires: int):
    device = init_nif_device(wires=n_wires)
    state_prep_op = qml.BasisState(np.zeros(n_wires, dtype=int), wires=range(n_wires))
    device.apply_state_prep(state_prep_op)
    return device


class TestGreedyStrategy:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize("n_wires", [2, 4])
    def test_returns_star_state_and_prob(self, n_wires):
        device = _setup_zero_state_device(n_wires)
        strategy = GreedyStrategy()
        states_prob_func = partial(device.get_states_probability, show_progress=False)
        star_states, star_probs = strategy(device, states_prob_func)

        assert star_states.shape[-1] == n_wires
        assert star_probs.shape == star_states.shape[:-1]

    def test_star_state_is_zero_for_zero_basis_state(self):
        n_wires = 2
        device = _setup_zero_state_device(n_wires)
        strategy = GreedyStrategy()
        states_prob_func = partial(device.get_states_probability, show_progress=False)
        star_states, star_probs = strategy(device, states_prob_func)

        np.testing.assert_array_equal(star_states.squeeze(), np.zeros(n_wires, dtype=int))
        np.testing.assert_allclose(star_probs.squeeze(), 1.0, atol=1e-6)
