import itertools

import numpy as np
import pytest
import pennylane as qml
import torch

from matchcake import utils
from matchcake.base.lookup_table import NonInteractingFermionicLookupTable
from matchcake.circuits import random_sptm_operations_generator
from scipy.linalg import expm
from torch.autograd import gradcheck
from ... import get_slow_test_mark
from ...test_nif_device import devices_init
from ...configs import (
    N_RANDOM_TESTS_PER_CASE,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed
)

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "input_matrix, system_state, target_binary_states",
    [
        (
                expm(np.random.randn(batch_size, 2 * size, 2 * size)),
                np.random.randint(2, size=size),
                np.random.randint(2, size=size),
        )
        for size in np.arange(1, 1+N_RANDOM_TESTS_PER_CASE, dtype=int)
        for batch_size in [1, 4]
    ]
)
def test_lookup_table_compute_observables_of_target_states_gradients(
        input_matrix, system_state, target_binary_states
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
        get_output, (init_params_nif, ),
        eps=1e-3,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON
    )


