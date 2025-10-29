import numpy as np
import pytest
import torch
from scipy.linalg import expm
from torch.autograd import gradcheck

from matchcake import utils
from matchcake.base.lookup_table import NonInteractingFermionicLookupTable
from matchcake.devices.probability_strategies import LookupTableStrategy
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)


@pytest.mark.parametrize(
    "batch_size, size",
    [
        (1, 3),
        (1, 6),
        (3, 4),
    ],
)
class TestNonInteractingFermionicLookupTableGradients:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.fixture
    def input_matrix(self, batch_size, size):
        return expm(np.random.randn(batch_size, 2 * size, 2 * size))

    @pytest.fixture
    def target_binary_states(self, size):
        return np.random.choice([0, 1], size=size)

    @pytest.fixture
    def system_state(self, size):
        return np.random.choice([0, 1], size=size)

    def test_compute_observables_of_target_states_gradients(
            self, input_matrix, system_state, target_binary_states
    ):
        def get_output(transition_matrix):
            lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
            batch_obs = lookup_table.compute_observables_of_target_states(
                system_state,
                target_binary_states,
                show_progress=False,
            )
            return batch_obs

        input_matrix = utils.make_transition_matrix_from_action_matrix(input_matrix)
        init_params_nif = torch.from_numpy(input_matrix).requires_grad_()
        assert gradcheck(
            get_output,
            (init_params_nif,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_compute_observables_of_target_states_gradients_lookup_table_strategy(self, input_matrix, system_state,
                                                                                  target_binary_states):
        def get_output(transition_matrix):
            lookup_table = NonInteractingFermionicLookupTable(transition_matrix)
            lookup_table_strategy = LookupTableStrategy()
            probs = lookup_table_strategy.batch_call(
                system_state=system_state,
                target_binary_states=target_binary_states,
                batch_wires=None,
                pfaffian_method="det",
                lookup_table=lookup_table,
            )
            return probs

        input_matrix = utils.make_transition_matrix_from_action_matrix(input_matrix)
        init_params_nif = torch.from_numpy(input_matrix).requires_grad_()
        assert gradcheck(
            get_output,
            (init_params_nif,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
