import numpy as np
import pytest

from matchcake import utils
from matchcake.operations import fRXX, fRYY, fRZZ, FermionicRotation
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
        for rn_params in np.random.uniform(0.0, np.pi / 2, size=(N_RANDOM_TESTS_PER_CASE, 2))
        for adjoint in [True, False]
    ],
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
        nif_expval,
        qubit_expval,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "initial_binary_string, cls_params_wires_list",
    [
        (i_b_string, [(rot, rn_params, [0, 1]), (qml.adjoint(rot), rn_params, [0, 1])])
        for rot in [fRXX, fRYY, fRZZ]
        for i_b_string in ["00", "01", "10", "11"]
        for rn_params in np.random.uniform(0.0, np.pi / 2, size=(N_RANDOM_TESTS_PER_CASE, 2))
    ],
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
        nif_expval,
        qubit_expval,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "x, directions, taylor_terms",
    [
        [np.random.rand(2), directions, taylor_terms]
        for directions in ["XX", "YY", "ZZ"]
        for taylor_terms in range(10, 20)
        for _ in range(N_RANDOM_TESTS_PER_CASE)
    ],
)
def test_fermionic_rotations_gradient_isfinite(x, directions, taylor_terms):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")

    params = torch.from_numpy(x).requires_grad_(True)
    FermionicRotation.USE_EXP_TAYLOR_SERIES = taylor_terms > 0
    FermionicRotation.TAYLOR_SERIES_TERMS = taylor_terms
    gate = FermionicRotation(params, wires=[0, 1], directions=directions)
    gate_real_mean = torch.real(torch.mean(gate.matrix()))
    gate_real_mean.backward()
    assert torch.all(
        torch.isfinite(params.grad)
    ), f"The gradient is not computed correctly for {directions}, {taylor_terms} terms and {x}."


@pytest.mark.parametrize(
    "x, directions, taylor_terms",
    [
        [np.array([0.12429722, 0.73086748]), "ZZ", 18],
        [np.array([0.12429722, 0.73086748]), "ZZ", 8],
    ],
)
def test_fermionic_rotations_gradient_isfinite_specific(x, directions, taylor_terms):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")

    torch.autograd.set_detect_anomaly(True)

    params = torch.from_numpy(x).requires_grad_(True)
    FermionicRotation.USE_EXP_TAYLOR_SERIES = True
    FermionicRotation.TAYLOR_SERIES_TERMS = taylor_terms
    gate = FermionicRotation(params, wires=[0, 1], directions=directions)
    gate_real_mean = torch.real(torch.mean(gate.matrix()))
    gate_real_mean.backward()
    assert torch.all(
        torch.isfinite(params.grad)
    ), f"The gradient is not computed correctly for {directions}, {taylor_terms} terms and {x}."
