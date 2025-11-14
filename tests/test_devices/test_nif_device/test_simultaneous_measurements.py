import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake import matchgate_parameter_sets as mgp
from matchcake import utils

from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)
from .. import devices_init, specific_matchgate_circuit


class TestNIFDeviceProbabilities:
    def test_fswap_probabilities_explicitsum(self):
        initial_binary_string = "01"
        params = mgp.fSWAP
        wires = [0, 1]
        target_binary_state = "10"
        prob = 1.0

        initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
        device = NonInteractingFermionicDevice(wires=wires, prob_strategy="ExplicitSum")
        operations = [
            qml.BasisState(initial_binary_state, wires=device.wires),
            MatchgateOperation(params, wires=wires),
        ]
        device.apply(operations)
        es_m_prob = device.get_state_probability(target_binary_state=target_binary_state, wires=wires)
        np.testing.assert_allclose(
            es_m_prob.squeeze(),
            prob,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "num_gates, num_wires, n_probs, prob_strategy",
        [
            (num_gates, num_wires, n_probs, "LookupTable")
            for num_wires in [2, 3, 4, 5, 6]
            for num_gates in [1, 3, 6]
            for n_probs in [1, num_wires]
        ]
        + [(3, 2, 2, "ExplicitSum")],
    )
    def test_multiples_matchgate_probs_against_qubit_device(self, num_gates, num_wires, n_probs, prob_strategy):
        params_list = [MatchgateOperation.random_params(seed=i) for i in range(num_gates)]
        prob_wires = np.random.choice(num_wires, replace=False, size=n_probs)

        nif_device, qubit_device = devices_init(wires=num_wires, prob_strategy=prob_strategy)
        nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
        qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

        all_wires = np.arange(num_wires)
        initial_binary_state = np.zeros(num_wires, dtype=int)
        wire0_vector = np.random.choice(all_wires[:-1], size=len(params_list))
        wire1_vector = wire0_vector + 1
        params_wires_list = [
            (params, [wire0, wire1]) for params, wire0, wire1 in zip(params_list, wire0_vector, wire1_vector)
        ]
        qubit_state = qubit_qnode(
            params_wires_list,
            initial_binary_state,
            all_wires=qubit_device.wires,
            in_param_type=mgp.MatchgatePolarParams,
            out_op="state",
        )
        qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
        nif_probs = nif_qnode(
            params_wires_list,
            initial_binary_state,
            all_wires=nif_device.wires,
            in_param_type=mgp.MatchgatePolarParams,
            out_op="probs",
            out_wires=prob_wires,
        )
        np.testing.assert_allclose(
            nif_probs.squeeze(),
            qubit_probs.squeeze(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
