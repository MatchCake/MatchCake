import numpy as np
import pytest

from matchcake import utils
from matchcake.operations import fRXX, fRYY, fRZZ
import pennylane as qml

from ..configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    set_seed,
    TEST_SEED,
)
from ..test_nif_device import devices_init
from . import specific_ops_circuit

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "initial_binary_string, cls_params_wires_list, is_adjoint",
    [
        (i_b_string, [(rot, rn_params, [0, 1])], adjoint)
        for rot in [fRXX, fRYY, fRZZ]
        for i_b_string in ["00", "01", "10", "11"]
        for rn_params in np.random.uniform(0.0, np.pi/2, size=(N_RANDOM_TESTS_PER_CASE, 2))
        for adjoint in [True, False]
    ]
)
def test_frot_in_circuit_with_pennylane(initial_binary_string, cls_params_wires_list, is_adjoint):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_ops_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_ops_circuit, qubit_device)

    qubit_expval = qubit_qnode(
        cls_params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        out_op="expval",
        adjoint=is_adjoint,
    )
    nif_expval = nif_qnode(
        cls_params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        out_op="expval",
        adjoint=is_adjoint,
    )
    np.testing.assert_allclose(
        nif_expval, qubit_expval,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_string, cls_params_wires_list",
    [
        (i_b_string, [(rot, rn_params, [0, 1]), (qml.adjoint(rot), rn_params, [0, 1])])
        for rot in [
            fRXX,
            fRYY,
            fRZZ
        ]
        for i_b_string in ["00", "01", "10", "11"]
        for rn_params in np.random.uniform(0.0, np.pi/2, size=(N_RANDOM_TESTS_PER_CASE, 2))
    ]
)
def test_frot_adj_circuit_with_pennylane(initial_binary_string, cls_params_wires_list):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_ops_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_ops_circuit, qubit_device)

    qubit_expval = qubit_qnode(
        cls_params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        out_op="expval",
    )
    nif_expval = nif_qnode(
        cls_params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        out_op="expval",
    )
    np.testing.assert_allclose(
        nif_expval, qubit_expval,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
