from typing import Tuple

import numpy as np
import pytest
import pennylane as qml
from msim import MatchgateOperator, NonInteractingFermionicDevice, Matchgate
from msim import matchgate_parameter_sets as mps
from msim import utils
from functools import partial

from msim.utils import PAULI_Z, PAULI_X
from . import devices_init
from ..configs import N_RANDOM_TESTS_PER_CASE

np.random.seed(42)


@pytest.mark.parametrize(
    "params,target_expectation_value",
    [
        (mps.Identity, np.array([1.0, 0.0])),
        (mps.MatchgateStandardParams(a=-1, w=-1, z=-1, d=-1), np.array([1.0, 0.0])),
        (mps.MatchgatePolarParams(r0=0, r1=1, theta1=0), np.array([0.0, 1.0])),
    ]
)
def test_single_gate_circuit_analytic_probability(params, target_expectation_value):
    device = NonInteractingFermionicDevice(wires=2)
    op = MatchgateOperator(params, wires=[0, 1])
    device.apply(op)
    expectation_value = device.analytic_probability(0)
    check = np.allclose(expectation_value, target_expectation_value)
    assert check, (f"The expectation value is not the correct one. "
                   f"Got {expectation_value} instead of {target_expectation_value}")


def single_matchgate_circuit(params, initial_state=np.array([0, 0])):
    qml.BasisState(initial_state, wires=[0, 1])
    MatchgateOperator(params, wires=[0, 1])
    return qml.probs(wires=0)


@pytest.mark.parametrize(
    "params,initial_binary_state",
    # [
    #     mps.MatchgateComposedHamiltonianParams()
    # ]
    # +
    # [
    #     mps.MatchgatePolarParams(r0=1, r1=1),
    #     mps.MatchgatePolarParams(r0=0, r1=1, theta1=0)
    # ]
    # +
    [
        # mps.MatchgatePolarParams(r0=1, r1=0, theta0=0, theta1=0, theta2=np.pi / 2, theta3=0, theta4=np.pi / 2),
        # (mps.fSWAP, np.array([0, 0])),
        # (mps.fSWAP, np.array([0, 1])),
        (mps.MatchgatePolarParams(
            r0=0.5, r1=0.1, theta0=np.pi, theta1=np.pi / 2, theta2=np.pi / 3, theta3=np.pi / 4, theta4=np.pi / 5
        ), np.array([0, 0])),
    ]
    # +
    # [
    #     mps.MatchgateHamiltonianCoefficientsParams(
    #         *np.random.rand(mps.MatchgateHamiltonianCoefficientsParams.N_PARAMS-1),
    #         epsilon=0.0
    #     )
    #     for _ in range(N_RANDOM_TESTS_PER_CASE)
    # ]
    # +
    # [
    #     mps.MatchgatePolarParams.random()
    #     for _ in range(N_RANDOM_TESTS_PER_CASE)
    # ]
    # +
    # [
    #     mps.MatchgatePolarParams(
    #         r0=np.random.uniform(0.09, 0.13),
    #         r1=np.random.uniform(0.0, 1.0),
    #         theta0=np.random.uniform(-np.pi, np.pi)*2,
    #         theta1=np.random.uniform(-np.pi, np.pi)*2,
    #         theta2=np.random.uniform(-np.pi, np.pi)*2,
    #         theta3=np.random.uniform(-np.pi, np.pi)*2,
    #     )
    #     for _ in range(N_RANDOM_TESTS_PER_CASE)
    # ]
)
def test_single_matchgate_probs_with_qbit_device(params, initial_binary_state):
    nif_device, qubit_device = devices_init()

    #
    device = NonInteractingFermionicDevice(wires=len(initial_binary_state))
    operations = [
        qml.BasisState(initial_binary_state, wires=device.wires),
        MatchgateOperator(params, wires=[0, 1])
    ]
    device.apply(operations)
    lt_probs = device.compute_probability_using_lookup_table(0)
    es_probs = device.compute_probability_using_explicit_sum(0)
    #

    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)

    qubit_probs = qubit_qnode(mps.MatchgatePolarParams.parse_from_any(params).to_numpy(), initial_binary_state)
    # nif_probs = nif_qnode(mps.MatchgatePolarParams.parse_from_any(params).to_numpy(), initial_binary_state)
    pred_probs = es_probs

    # TODO: to remove
    TBT = device.transition_matrix.conj() @ device.lookup_table.block_diagonal_matrix @ device.transition_matrix.T
    obs = np.array([
        [0, TBT[0, 0]],
        [-TBT[0, 0], 0]
    ])
    from pfapack import pfaffian
    p1 = pfaffian.pfaffian(obs)
    #

    same_argmax = np.argmax(pred_probs) == np.argmax(qubit_probs)
    assert same_argmax, (f"The argmax is not the correct one. "
                         f"Got {np.argmax(pred_probs)} instead of {np.argmax(qubit_probs)}")
    check = np.allclose(pred_probs, qubit_probs, rtol=1.e-1, atol=1.e-1)
    assert check, f"The probs are not the correct one. Got {pred_probs} instead of {qubit_probs}"
