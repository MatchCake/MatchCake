import numpy as np
import pennylane as qml
import pytest

from matchcake import utils
from matchcake.operations import MAngleEmbedding

from ..configs import (ATOL_APPROX_COMPARISON, N_RANDOM_TESTS_PER_CASE,
                       RTOL_APPROX_COMPARISON, TEST_SEED, set_seed)
from ..test_nif_device import devices_init
from . import specific_ops_circuit

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "initial_binary_string, cls_params_wires_list, is_adjoint",
    [
        (
            i_b_string,
            [(MAngleEmbedding, rn_params, [0, 1], {"rotations": rot})],
            adjoint,
        )
        for rot in ["X", "Y", "Z", "X,Y", "X,Z", "Y,Z", "X,Y,Z"]
        for i_b_string in ["00", "01", "10", "11"]
        for rn_params in np.random.uniform(0.0, np.pi / 2, size=(N_RANDOM_TESTS_PER_CASE, 2))
        for adjoint in [True, False]
    ],
)
def test_m_angle_embedding_with_pennylane(initial_binary_string, cls_params_wires_list, is_adjoint):
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
        nif_expval,
        qubit_expval,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
