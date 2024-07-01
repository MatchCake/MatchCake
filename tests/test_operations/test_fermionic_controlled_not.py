import numpy as np
import pytest

from matchcake import utils
from matchcake.operations import fCNOT
import pennylane as qml

from ..configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    TEST_SEED,
)
from ..test_nif_device import devices_init
from . import specific_ops_circuit

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "initial_binary_string, cls_params_wires_list, adjoint",
    [
        (i_b_string, [(fCNOT, [0, 1])], adjoint)
        for i_b_string in ["00", "01", "10", "11"]
        for adjoint in [True, False]
    ]
)
def test_fcnot_in_circuit_with_pennylane(initial_binary_string, cls_params_wires_list, adjoint):
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state))

    nif_qnode = qml.QNode(specific_ops_circuit, nif_device)
    qubit_qnode = qml.QNode(specific_ops_circuit, qubit_device)

    qubit_expval = qubit_qnode(
        cls_params_wires_list,
        initial_binary_state,
        all_wires=qubit_device.wires,
        out_op="expval",
        adjoint=adjoint,
    )
    nif_expval = nif_qnode(
        cls_params_wires_list,
        initial_binary_state,
        all_wires=nif_device.wires,
        out_op="expval",
        adjoint=adjoint,
    )
    np.testing.assert_allclose(
        nif_expval, qubit_expval,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
