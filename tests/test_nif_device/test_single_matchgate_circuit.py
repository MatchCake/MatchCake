import numpy as np
import pennylane as qml
import pytest

from msim import MatchgateOperation, NonInteractingFermionicDevice
from msim import matchgate_parameter_sets as mps
from msim import utils
from ..configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
)

np.random.seed(TEST_SEED)


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
    op = MatchgateOperation(params, wires=[0, 1])
    device.apply(op)
    expectation_value = device.analytic_probability(0)
    np.testing.assert_allclose(
        expectation_value, target_expectation_value,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


def single_matchgate_circuit(params, initial_state=np.array([0, 0]), **kwargs):
    qml.BasisState(initial_state, wires=[0, 1])
    MatchgateOperation(params, wires=[0, 1], **kwargs)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@pytest.mark.parametrize(
    "params,initial_binary_state",
    [
        (mps.MatchgateComposedHamiltonianParams(), [0, 0])
    ]
    +
    [
        (mps.MatchgatePolarParams(r0=1, r1=1), [0, 0]),
        (mps.MatchgatePolarParams(r0=0, r1=1, theta1=0), [0, 0]),
    ]
    +
    [
        (mps.MatchgatePolarParams(r0=1, theta2=np.pi / 2, theta4=np.pi / 2), [0, 0]),
        (mps.fSWAP, [0, 0]),
        (mps.fSWAP, [0, 1]),
        (mps.HellParams, [0, 0]),
    ]
    +
    [
        (mps.MatchgatePolarParams.random(), i_state)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for i_state in [[0, 0], [0, 1], [1, 0], [1, 1]]
    ]
    +
    [
        (mps.MatchgateComposedHamiltonianParams.random(), i_state)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for i_state in [[0, 0], [0, 1], [1, 0], [1, 1]]
    ]
    +
    [
        (mps.MatchgatePolarParams(
            r0=np.random.uniform(0.09, 0.13),
            r1=np.random.uniform(0.0, 1.0),
            theta0=np.random.uniform(-np.pi, np.pi) * 2,
            theta1=np.random.uniform(-np.pi, np.pi) * 2,
            theta2=np.random.uniform(-np.pi, np.pi) * 2,
            theta3=np.random.uniform(-np.pi, np.pi) * 2,
        ), i_state)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for i_state in [[0, 0], [0, 1], [1, 0], [1, 1]]
    ]
)
def test_single_matchgate_probs_with_qbit_device(params, initial_binary_state):
    from . import devices_init
    nif_device, qubit_device = devices_init(prob_strategy="explicit_sum")
    initial_binary_state = np.asarray(initial_binary_state)
    
    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    prob_wires = 0
    qubit_state = qubit_qnode(
        mps.MatchgatePolarParams.parse_from_any(params).to_numpy(),
        initial_binary_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
    nif_probs = nif_qnode(
        mps.MatchgatePolarParams.parse_from_any(params).to_numpy(),
        initial_binary_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=prob_wires,
    )
    
    np.testing.assert_allclose(
        nif_probs, qubit_probs,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
