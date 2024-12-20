import itertools
from functools import partial

import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.ops.qubit.observables import BasisStateProjector
import pytest

from matchcake import MatchgateOperation, utils
from matchcake import matchgate_parameter_sets as mps
from matchcake import operations
from matchcake.utils.torch_utils import to_numpy
from . import devices_init
from ..configs import (
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    N_RANDOM_TESTS_PER_CASE,
)

set_seed(TEST_SEED)


__states_and_gates = [
    ("00", [(mps.fSWAP, [0, 1])]),
    ("01", [(mps.fSWAP, [0, 1])]),
    ("10", [(mps.fSWAP, [0, 1])]),
    ("11", [(mps.fSWAP, [0, 1])]),
    ("00", [(mps.HellParams, [0, 1])]),
    ("01", [(mps.HellParams, [0, 1])]),
    ("10", [(mps.HellParams, [0, 1])]),
    ("11", [(mps.HellParams, [0, 1])]),
    ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3])]),
    ("000000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3]), (mps.fSWAP, [4, 5])]),
    ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.fSWAP, [2, 3])]),
    ("0000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3])]),
    ("000000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3]), (mps.fSWAP, [4, 5])]),
    ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.HellParams, [2, 3])]),
]


def specific_matchgate_circuit(params_wires_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    use_h_for_transition_matrix = kwargs.pop(
        "use_h_for_transition_matrix", MatchgateOperation.DEFAULT_USE_H_FOR_TRANSITION_MATRIX
    )
    if all_wires is None:
        all_wires = set(sum([list(wires) for _, wires in params_wires_list], []))
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    for params, wires in params_wires_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        MatchgateOperation(mg_params, wires=wires, use_h_for_transition_matrix=use_h_for_transition_matrix)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    elif out_op == "expval":
        projector: BasisStateProjector = qml.Projector(initial_state, wires=all_wires)
        return qml.expval(projector)
    elif out_op == "sample":
        return qml.sample(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


@pytest.mark.parametrize(
    "initial_binary_string,params_wires_list,contraction_strategy",
    [
        (initial_binary_string, params_wires_list, contraction_strategy)
        for [initial_binary_string, params_wires_list], contraction_strategy in
        list(itertools.product(*[__states_and_gates, [
            None,
            "neighbours",
            "forward",
            "horizontal",
            "vertical"
        ]]))
    ]
)
def test_multiples_matchgate_probs_with_qbit_device(initial_binary_string, params_wires_list, contraction_strategy):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state), contraction_strategy=contraction_strategy)

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    qubit_probs = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=qubit_device.wires,
    )
    nif_probs = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        out_wires=nif_device.wires,
    )
    qubit_probs = to_numpy(qubit_probs).squeeze()
    nif_probs = to_numpy(nif_probs).squeeze()
    np.testing.assert_allclose(
        nif_probs.squeeze(), qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_string,params_wires_list",
    [
        ("00", [(mps.fSWAP, [0, 1])]),
        ("01", [(mps.fSWAP, [0, 1])]),
        ("10", [(mps.fSWAP, [0, 1])]),
        ("11", [(mps.fSWAP, [0, 1])]),
        ("00", [(mps.HellParams, [0, 1])]),
        ("01", [(mps.HellParams, [0, 1])]),
        ("10", [(mps.HellParams, [0, 1])]),
        ("11", [(mps.HellParams, [0, 1])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3])]),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3]), (mps.fSWAP, [4, 5])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.fSWAP, [2, 3])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3])]),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3]), (mps.fSWAP, [4, 5])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.HellParams, [2, 3])]),
    ]
)
def test_multiples_matchgate_expval_with_qbit_device(initial_binary_string, params_wires_list):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    qubit_expval = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="expval",
    )
    nif_expval = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="expval",
    )
    np.testing.assert_allclose(
        nif_expval.squeeze(), qubit_expval.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_string,params_wires_list",
    [
        ("00", [(mps.fSWAP, [0, 1])]),
        ("01", [(mps.fSWAP, [0, 1])]),
        ("10", [(mps.fSWAP, [0, 1])]),
        ("11", [(mps.fSWAP, [0, 1])]),
        ("00", [(mps.HellParams, [0, 1])]),
        ("01", [(mps.HellParams, [0, 1])]),
        ("10", [(mps.HellParams, [0, 1])]),
        ("11", [(mps.HellParams, [0, 1])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3])]),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3]), (mps.fSWAP, [4, 5])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.fSWAP, [2, 3])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3])]),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3]), (mps.fSWAP, [4, 5])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.HellParams, [2, 3])]),
    ]
)
def test_multiples_matchgate_expval_with_qubit_device_with_h_transition(initial_binary_string, params_wires_list):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    qubit_expval = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="expval",
        use_h_for_transition_matrix=False,
    )
    nif_expval = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="expval",
        use_h_for_transition_matrix=True,
    )
    np.testing.assert_allclose(
        nif_expval.squeeze(), qubit_expval.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_string,params_wires_list",
    [
        ("00", [(mps.fSWAP, [0, 1])]),
        ("01", [(mps.fSWAP, [0, 1])]),
        ("10", [(mps.fSWAP, [0, 1])]),
        ("11", [(mps.fSWAP, [0, 1])]),
        ("00", [(mps.HellParams, [0, 1])]),
        ("01", [(mps.HellParams, [0, 1])]),
        ("10", [(mps.HellParams, [0, 1])]),
        ("11", [(mps.HellParams, [0, 1])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3])]),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [2, 3]), (mps.fSWAP, [4, 5])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.fSWAP, [2, 3])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3])]),
        ("000000", [(mps.fSWAP, [0, 1]), (mps.HellParams, [2, 3]), (mps.fSWAP, [4, 5])]),
        ("0000", [(mps.fSWAP, [0, 1]), (mps.fSWAP, [1, 2]), (mps.HellParams, [2, 3])]),
    ]
)
def test_multiples_matchgate_probs_with_qubit_device_with_h_transition(initial_binary_string, params_wires_list):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_matchgate_circuit, qubit_device)

    qubit_expval = qubit_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        use_h_for_transition_matrix=False,
    )
    nif_expval = nif_qnode(
        params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        in_param_type=mps.MatchgatePolarParams,
        out_op="probs",
        use_h_for_transition_matrix=True,
    )
    np.testing.assert_allclose(
        nif_expval.squeeze(), qubit_expval.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "theta,contraction_strategy",
    [
        (theta, contraction_strategy)
        for theta in np.linspace(0, 2 * np.pi, num=N_RANDOM_TESTS_PER_CASE)
        for contraction_strategy in [None, "neighbours", "forward", "horizontal", "vertical"]
    ]
)
def test_multiples_matchgate_state_with_qbit_device_zyz(theta, contraction_strategy):
    initial_binary_string = "00"
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state), contraction_strategy=contraction_strategy)

    def circuit_state():
        operations.fRZZ(np.asarray([theta, theta]), wires=[0, 1])
        operations.fRYY(np.asarray([theta, theta]), wires=[0, 1])
        operations.fRZZ(np.asarray([theta, theta]), wires=[0, 1])
        return qml.state()

    def circuit_probs():
        operations.fRZZ(np.asarray([theta, theta]), wires=[0, 1])
        operations.fRYY(np.asarray([theta, theta]), wires=[0, 1])
        operations.fRZZ(np.asarray([theta, theta]), wires=[0, 1])
        return qml.probs()

    qubit_state = qml.QNode(circuit_state, qubit_device)()
    expected_state = np.asarray([np.exp(-1j*theta)*np.cos(theta/2), 0, 0, np.sin(theta/2)])

    np.testing.assert_allclose(
        qubit_state.squeeze(), expected_state.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    qubit_probs = qml.QNode(circuit_probs, qubit_device)()
    nif_probs = qml.QNode(circuit_probs, nif_device)()
    qubit_probs = to_numpy(qubit_probs).squeeze()
    nif_probs = to_numpy(nif_probs).squeeze()

    np.testing.assert_allclose(
        nif_probs.squeeze(), qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


