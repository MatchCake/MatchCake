import numpy as np
import pennylane as qml
import pytest

from matchcake import matchgate_parameter_sets as mps
from matchcake.circuits import random_sptm_operations_generator
from matchcake.operations import SptmCompRxRx
from matchcake.utils import torch_utils

from ..configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from . import devices_init
from .test_specific_circuit import specific_matchgate_circuit


class TestNonInteractingFermionicDeviceSampling:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "num_gates, num_wires",
        [
            (10, 2),
            (30, 4),
            (10, 5),
            (10, 6),
        ],
    )
    def test_qubit_by_qubit_sampling_with_probs(self, num_gates, num_wires):
        params_list = [mps.MatchgatePolarParams.random().to_numpy() for _ in range(num_gates)]

        nif_device, _ = devices_init(
            wires=num_wires,
            shots=int(1024 * num_wires),
            sampling_strategy="QubitByQubitSampling",
        )
        nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)

        all_wires = np.arange(num_wires)
        initial_binary_state = np.zeros(num_wires, dtype=int)
        wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
        wire1_vector = wire0_vector + 1
        params_wires_list = [
            (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
        ]
        nif_samples = torch_utils.to_numpy(
            nif_qnode(
                params_wires_list,
                initial_binary_state,
                all_wires=nif_device.wires,
                in_param_type=mps.MatchgatePolarParams,
                out_op="sample",
            )
        ).astype(int)
        states = np.unique(nif_samples, axis=0).astype(int)
        states_probability = np.array(
            [np.sum(np.all(nif_samples == state, axis=1)) / nif_samples.shape[0] for state in states]
        )
        states_probability = states_probability / np.sum(states_probability)
        states_expval = np.array([nif_device.get_state_probability(target_binary_state=state) for state in states])
        states_expval = states_expval / np.sum(states_expval)

        abs_diff = np.abs(states_probability - states_expval)
        np.testing.assert_allclose(
            states_probability.squeeze(),
            states_expval.squeeze(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"abs_diff: {abs_diff.tolist()}",
        )

    @pytest.mark.parametrize(
        "batch_size, num_gates, num_wires",
        [
            (batch_size, num_gates, num_wires)
            for num_wires in range(2, 6)
            for num_gates in [1, 10 * num_wires]
            for batch_size in [None, 16]
        ],
    )
    def test_qubit_by_qubit_sampling_with_probs_op_gen(self, batch_size, num_gates, num_wires):
        operations_generator = random_sptm_operations_generator(
            num_gates,
            np.arange(num_wires),
            batch_size=batch_size,
            op_types=[SptmCompRxRx],
        )
        nif_device, _ = devices_init(
            wires=num_wires,
            shots=int(1024 * num_wires),
            sampling_strategy="QubitByQubitSampling",
        )
        nif_samples = torch_utils.to_numpy(
            nif_device.execute_generator(operations_generator, output_type="samples")
        ).astype(int)

        unique_states = np.unique(nif_samples.reshape(-1, num_wires), axis=0).astype(int)
        unique_states_probability = np.stack(
            [
                np.sum(np.isclose(nif_samples, state).all(axis=-1), axis=0) / nif_samples.shape[0]
                for state in unique_states
            ],
            axis=0,
        )
        unique_states_probability = unique_states_probability / np.sum(unique_states_probability, axis=0, keepdims=True)
        unique_states_expval = nif_device.get_states_probability(unique_states, np.arange(num_wires))
        states_expval = unique_states_expval / np.sum(unique_states_expval, axis=0, keepdims=True)

        abs_diff = np.abs(unique_states_probability - states_expval)
        np.testing.assert_allclose(
            unique_states_probability.squeeze(),
            states_expval.squeeze(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"abs_diff: {abs_diff.tolist()}",
        )

    @pytest.mark.parametrize(
        "batch_size, num_gates, num_wires",
        [
            (batch_size, num_gates, num_wires)
            for num_wires in range(2, 6)
            for num_gates in [1, 10 * num_wires]
            for batch_size in [None, 16]
        ],
    )
    def test_2qubits_by_2qubits_sampling_with_probs_op_gen(self, batch_size, num_gates, num_wires):
        operations_generator = random_sptm_operations_generator(
            num_gates,
            np.arange(num_wires),
            batch_size=batch_size,
            op_types=[SptmCompRxRx],
        )

        nif_device, _ = devices_init(
            wires=num_wires,
            shots=int(2048 * num_wires),
            sampling_strategy="2QubitBy2QubitSampling",
        )
        nif_samples = torch_utils.to_numpy(
            nif_device.execute_generator(operations_generator, output_type="samples")
        ).astype(int)

        unique_states = np.unique(nif_samples.reshape(-1, num_wires), axis=0).astype(int)
        unique_states_probability = np.stack(
            [
                np.sum(np.isclose(nif_samples, state).all(axis=-1), axis=0) / nif_samples.shape[0]
                for state in unique_states
            ],
            axis=0,
        )
        unique_states_probability = unique_states_probability / np.sum(unique_states_probability, axis=0, keepdims=True)
        unique_states_expval = nif_device.get_states_probability(unique_states, np.arange(num_wires))
        states_expval = unique_states_expval / np.sum(unique_states_expval, axis=0, keepdims=True)

        abs_diff = np.abs(unique_states_probability - states_expval)
        np.testing.assert_allclose(
            unique_states_probability.squeeze(),
            states_expval.squeeze(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg=f"abs_diff: {abs_diff.tolist()}",
        )
