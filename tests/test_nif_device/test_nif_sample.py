import numpy as np
import pennylane as qml
import pytest
import psutil

from matchcake import MatchgateOperation, utils
from matchcake.utils import torch_utils
from matchcake import matchgate_parameter_sets as mps
from . import devices_init
from .test_specific_circuit import specific_matchgate_circuit
from .. import get_slow_test_mark
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)

set_seed(TEST_SEED)


@get_slow_test_mark()
@pytest.mark.slow
@pytest.mark.parametrize(
    "params_list,n_wires",
    [
        ([mps.MatchgatePolarParams.random()], 2),
        ([mps.MatchgatePolarParams.random(), mps.MatchgatePolarParams.random()], 2),
        ([mps.MatchgatePolarParams.random(), mps.MatchgatePolarParams.random()], 4)
    ]
    +
    [
        ([mps.MatchgatePolarParams(r0=1, r1=1).to_numpy() for _ in range(num_gates)], num_wires)
        for num_wires in range(2, 6)
        for num_gates in [1, 2 ** num_wires]
    ]
    +
    [
        ([mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)], num_wires)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for num_wires in range(2, 6)
        for num_gates in [1, 2 ** num_wires]
    ]
)
def test_multiples_matchgate_probs_with_qbit_device(params_list, n_wires):
    nif_device, _ = devices_init(wires=n_wires, shots=int(1024 * n_wires))
    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)

    all_wires = np.arange(n_wires)
    initial_binary_state = np.zeros(n_wires, dtype=int)
    wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
    wire1_vector = wire0_vector + 1
    params_wires_list = [
        (params, [wire0, wire1])
        for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
    ]
    nif_samples = torch_utils.to_numpy(nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="sample"
    )).astype(int)
    states = np.unique(nif_samples, axis=0).astype(int)
    states_probability = np.array([
        np.sum(np.all(nif_samples == state, axis=1)) / nif_samples.shape[0]
        for state in states
    ])
    states_probability = states_probability / np.sum(states_probability)
    states_expval = np.array([
        nif_device.get_state_probability(target_binary_state=state)
        for state in states
    ])
    states_expval = states_expval / np.sum(states_expval)

    abs_diff = np.abs(states_probability - states_expval)
    np.testing.assert_allclose(
        states_probability.squeeze(), states_expval.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
        err_msg=f"abs_diff: {abs_diff.tolist()}"
    )
