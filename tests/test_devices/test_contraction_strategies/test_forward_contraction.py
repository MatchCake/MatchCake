import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake import MatchgateOperation
from matchcake import utils
from matchcake.operations import SptmCompRxRx, SptmIdentity, MatchgateIdentity
from matchcake.utils import torch_utils
from matchcake import matchgate_parameter_sets as mgp

from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .. import devices_init, init_nif_device
from ..test_single_line_matchgates_circuit import single_line_matchgates_circuit


class TestNonInteractingFermionicDeviceForwardContractionStrategy:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize(
        "operations,expected_new_operations",
        [
            (
                [MatchgateIdentity(wires=[0, 1])],
                [MatchgateIdentity(wires=[0, 1])],
            ),
        ],
    )
    def test_forward_contraction(self, operations, expected_new_operations):
        all_wires = set(wire for op in operations for wire in op.wires)
        nif_device_nh = init_nif_device(wires=len(all_wires), contraction_method="forward")
        new_operations = nif_device_nh.contraction_strategy(operations)

        assert len(new_operations) == len(expected_new_operations), "The number of operations is different."
        for new_op, expected_op in zip(new_operations, expected_new_operations):
            np.testing.assert_allclose(
                new_op.matrix(),
                expected_op.matrix(),
                atol=ATOL_APPROX_COMPARISON,
                rtol=RTOL_APPROX_COMPARISON,
            )

    @pytest.mark.parametrize(
        "operations",
        [
            [MatchgateIdentity(wires=[0, 1])],
            [MatchgateIdentity(wires=[0, 1]) for _ in range(10)],
        ],
    )
    def test_forward_contraction_device_one_line_identity(self, operations):
        nif_device_nh = init_nif_device(wires=2, contraction_method="forward")
        nif_device = init_nif_device(wires=2, contraction_method=None)

        nif_device_nh.apply(operations)
        nif_device.apply(operations)

        np.testing.assert_allclose(
            nif_device.analytic_probability(),
            nif_device_nh.analytic_probability(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "batch_size, num_operations",
        [(10, 1), (5, 2), (2, 3)],
    )
    def test_forward_contraction_device_one_line_random(self, batch_size, num_operations):
        operations = [
            MatchgateOperation.random(batch_size=batch_size, wires=[0, 1])
            for _ in range(num_operations)
        ]
        nif_device_nh = init_nif_device(wires=2, contraction_method="forward")
        nif_device = init_nif_device(wires=2, contraction_method=None)

        nif_device_nh.apply(operations)
        nif_device.apply(operations)

        np.testing.assert_allclose(
            nif_device.analytic_probability(),
            nif_device_nh.analytic_probability(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "operations",
        [
            [SptmIdentity(wires=[0, 1])],
            [SptmIdentity(wires=[0, 1]) for _ in range(10)],
        ],
    )
    def test_forward_contraction_device_one_line_sptm_identity(self, operations):
        nif_device_nh = init_nif_device(wires=2, contraction_method="forward")
        nif_device = init_nif_device(wires=2, contraction_method=None)

        nif_device_nh.apply(operations)
        nif_device.apply(operations)

        np.testing.assert_allclose(
            nif_device.analytic_probability(),
            nif_device_nh.analytic_probability(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("num_operations", [1, 3, 10])
    def test_forward_contraction_device_one_line_sptm_random(self, num_operations):
        operations = [SptmCompRxRx(np.random.random(2), wires=[0, 1]) for _ in range(num_operations)]
        nif_device_nh = init_nif_device(wires=2, contraction_method="forward")
        nif_device = init_nif_device(wires=2, contraction_method=None)

        nif_device_nh.apply(operations)
        nif_device.apply(operations)

        np.testing.assert_allclose(
            nif_device.analytic_probability(),
            nif_device_nh.analytic_probability(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("num_gates", list(2 ** np.arange(1, 5)))
    def test_multiples_matchgate_probs_with_qbit_device_forward_contraction(self, num_gates):
        params_list = [MatchgateOperation.random_params(seed=i) for i in range(num_gates)]
        prob_wires = 0
        nif_device, qubit_device = devices_init(wires=2, contraction_method="forward")

        nif_qnode = qml.QNode(single_line_matchgates_circuit, nif_device)
        qubit_qnode = qml.QNode(single_line_matchgates_circuit, qubit_device)

        initial_binary_state = np.array([0, 0])
        qubit_state = qubit_qnode(
            params_list,
            initial_binary_state,
            in_param_type=mgp.MatchgatePolarParams,
            out_op="state",
        )
        qubit_probs = utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
        nif_probs = nif_qnode(
            params_list,
            initial_binary_state,
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

    def test_forward_contraction_torch_grad(self):
        x = np.random.rand(4)
        n_qubits = 4
        x = torch.from_numpy(x).float()
        x_grad = x.detach().clone().requires_grad_(True)

        dev = mc.NonInteractingFermionicDevice(wires=n_qubits, contraction_method="forward")

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            mc.operations.CompRyRy(x[0:2], wires=[0, 1])
            mc.operations.CompRyRy(x[2:4], wires=[2, 3])
            mc.operations.CompRyRy(x[0:2], wires=[0, 1])
            mc.operations.CompRyRy(x[2:4], wires=[2, 3])
            return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

        try:
            circuit(x_grad)
        except Exception as e:
            pytest.fail(f"Error during forward pass: {e}")

        np.testing.assert_allclose(
            torch_utils.to_numpy(circuit(x)),
            torch_utils.to_numpy(circuit(x_grad)),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
            err_msg="Forward pass with and without gradient computation are different.",
        )
