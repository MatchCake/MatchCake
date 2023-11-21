from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from pfapack import pfaffian

from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils
from functools import partial

from msim.utils import PAULI_Z, PAULI_X
from .configs import N_RANDOM_TESTS_PER_CASE
from .test_nif_device import single_matchgate_circuit

np.random.seed(42)

fSWAP_R = Matchgate(mps.fSWAP).single_transition_particle_matrix


@pytest.mark.parametrize(
    "params,initial_binary_state,output_state",
    [
        (mps.Identity, "00", np.array([1, 0, 0, 0])),
        (mps.Identity, "01", np.array([0, 1, 0, 0])),
        (mps.Identity, "10", np.array([0, 0, 1, 0])),
        (mps.Identity, "11", np.array([0, 0, 0, 1])),
        (mps.fSWAP, "00", np.array([1, 0, 0, 0])),
        (mps.fSWAP, "01", np.array([0, 0, 1, 0])),
        (mps.fSWAP, "10", np.array([0, 1, 0, 0])),
        (mps.fSWAP, "11", np.array([0, 0, 0, -1])),
        (mps.HellParams, "00", np.array([-0.5, 0, 0, np.sqrt(0.75)*1j])),
    ]
)
def test_single_matchgate_circuit_output_state(params, initial_binary_state, output_state):
    from .test_nif_device import devices_init
    _, qubit_device = devices_init()
    initial_state = utils.binary_string_to_vector(initial_binary_state)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    qubit_state = qubit_qnode(
        mps.MatchgatePolarParams.parse_from_any(params).to_numpy(),
        initial_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    assert np.allclose(qubit_state, output_state), (
        f"The output state is not the correct one. Got {qubit_state} instead of {output_state}."
    )
    
    
@pytest.mark.parametrize(
    "params,initial_binary_state,output_probs",
    [
        (mps.Identity, "00", np.array([1, 0, 0, 0])),
        (mps.Identity, "01", np.array([0, 1, 0, 0])),
        (mps.Identity, "10", np.array([0, 0, 1, 0])),
        (mps.Identity, "11", np.array([0, 0, 0, 1])),
        (mps.fSWAP, "00", np.array([1, 0, 0, 0])),
        (mps.fSWAP, "01", np.array([0, 0, 1, 0])),
        (mps.fSWAP, "10", np.array([0, 1, 0, 0])),
        (mps.fSWAP, "11", np.array([0, 0, 0, 1])),
        (mps.HellParams, "00", np.array([0.25, 0, 0, 0.75])),
    ]
)
def test_single_matchgate_circuit_output_probs(params, initial_binary_state, output_probs):
    from .test_nif_device import devices_init
    _, qubit_device = devices_init()
    initial_state = utils.binary_string_to_vector(initial_binary_state)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    qubit_state = qubit_qnode(
        mps.MatchgatePolarParams.parse_from_any(params).to_numpy(),
        initial_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state)
    assert np.allclose(qubit_probs, output_probs), (
        f"The output probs is not the correct one. Got {qubit_probs} instead of {output_probs}."
    )


@pytest.mark.parametrize(
    "params,k,binary_state,observable",
    [
        (
            mps.fSWAP, 0, "01",
            np.array([
                [0, 1, 0, 1],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
                [-1, 0, -1, 0]
            ])
        ),
        # (
        #     mps.MatchgatePolarParams(
        #         r0=1, r1=0, theta0=np.pi, theta1=np.pi / 2, theta2=np.pi / 3, theta3=np.pi / 4, theta4=np.pi / 5
        #     ),
        #     0, "01",
        #     np.array([
        #         [0, 1, 0, 1],
        #         [-1, 0, 0, 0],
        #         [0, 0, 0, 1],
        #         [-1, 0, -1, 0]
        #     ])
        # )
    ]
)
def test_single_matchgate_obs_on_specific_cases(params, k, binary_state, observable):
    nif_device = NonInteractingFermionicDevice(wires=2)
    nif_device.apply(MatchgateOperator(params, wires=[0, 1]))
    state = utils.binary_state_to_state(binary_state)
    pred_obs = nif_device.lookup_table.get_observable(k, state)
    pred_pf = pfaffian.pfaffian(pred_obs)
    pf = pfaffian.pfaffian(observable)
    assert np.allclose(pred_obs, observable), (
        f"The observable is not the correct one. Got \n{pred_obs} instead of \n{observable}"
    )
    assert np.allclose(pred_pf, pf), (
        f"The Pfaffian is not the correct one. Got \n{pred_pf} instead of \n{pf}"
    )


@pytest.mark.parametrize(
    "wires, n_wires, padded_matrix",
    [
        (
            [0, 1], 2, fSWAP_R,
        ),
    ]
)
def test_get_padded_single_transition_particle_matrix(wires, n_wires, padded_matrix):
    mgo = MatchgateOperator(mps.fSWAP, wires=wires)
    all_wires = list(range(n_wires))
    pred_padded_matrix = mgo.get_padded_single_transition_particle_matrix(wires=all_wires)
    assert np.allclose(pred_padded_matrix, padded_matrix), (
        f"The padded matrix is not the correct one. Got \n{pred_padded_matrix} instead of \n{padded_matrix}"
    )

