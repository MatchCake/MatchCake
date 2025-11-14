import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import utils
from matchcake.operations import CompRotation, CompRxRx, CompRyRy, CompRzRz

from ..configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from ..test_devices import devices_init
from . import specific_ops_circuit


class TestCompRotation:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "initial_binary_string, rot, is_adjoint, seed",
        [
            (i_b_string, rot, adjoint, seed)
            for rot in [CompRxRx, CompRyRy, CompRzRz]
            for i_b_string in ["00", "01", "10", "11"]
            for adjoint in [True, False]
            for seed in [0, 1, 2]
        ],
    )
    def test_frot_in_circuit_with_pennylane(self, initial_binary_string, rot, is_adjoint, seed):
        rn_params = rot.random_params(seed=seed)
        cls_params_wires_list = [(rot, rn_params, [0, 1])]

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
        "initial_binary_string, rot",
        [(i_b_string, rot) for rot in [CompRxRx, CompRyRy, CompRzRz] for i_b_string in ["00", "01", "10", "11"]],
    )
    def test_frot_adj_circuit_with_pennylane(self, initial_binary_string, rot):
        rn_params = np.random.uniform(0.0, np.pi / 2, size=2)
        cls_params_wires_list = [(rot, rn_params, [0, 1]), (qml.adjoint(rot), rn_params, [0, 1])]

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
        "directions, taylor_terms",
        [[directions, taylor_terms] for directions in ["XX", "YY", "ZZ"] for taylor_terms in range(10, 20)],
    )
    def test_fermionic_rotations_gradient_isfinite(self, directions, taylor_terms):
        x = np.random.rand(2)

        params = torch.from_numpy(x).requires_grad_(True)
        CompRotation.USE_EXP_TAYLOR_SERIES = taylor_terms > 0
        CompRotation.TAYLOR_SERIES_TERMS = taylor_terms
        gate = CompRotation(params, wires=[0, 1], directions=directions)
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
    def test_fermionic_rotations_gradient_isfinite_specific(self, x, directions, taylor_terms):
        params = torch.from_numpy(x).requires_grad_(True)
        CompRotation.USE_EXP_TAYLOR_SERIES = True
        CompRotation.TAYLOR_SERIES_TERMS = taylor_terms
        gate = CompRotation(params, wires=[0, 1], directions=directions)
        gate_real_mean = torch.real(torch.mean(gate.matrix()))
        gate_real_mean.backward()
        assert torch.all(
            torch.isfinite(params.grad)
        ), f"The gradient is not computed correctly for {directions}, {taylor_terms} terms and {x}."
