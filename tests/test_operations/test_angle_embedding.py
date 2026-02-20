import numpy as np
import pennylane as qml
import pytest

from matchcake import utils
from matchcake.operations import MAngleEmbedding, MAngleEmbeddings

from ..configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from ..test_devices import devices_init
from . import specific_ops_circuit


class TestMAngleEmbedding:
    @pytest.mark.parametrize(
        "initial_binary_string, rot, is_adjoint, seed",
        [
            (i_b_string, rot, adjoint, seed)
            for rot in ["X", "Y", "Z", "X,Y", "X,Z", "Y,Z", "X,Y,Z"]
            for i_b_string in ["00", "01", "10", "11"]
            for adjoint in [True, False]
            for seed in range(10)
        ],
    )
    def test_m_angle_embedding_with_pennylane_rn_params(
        self,
        initial_binary_string,
        rot,
        is_adjoint,
        seed,
    ):
        rn_state = np.random.RandomState(seed=seed)
        rn_params = rn_state.uniform(0.0, np.pi / 2, size=(2,))
        cls_params_wires_list = [(MAngleEmbedding, rn_params, [0, 1], {"rotations": rot})]
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

    def test_repr(self):
        op = MAngleEmbedding(np.asarray([0.1, 0.2]), [0, 1])
        op_repr = repr(op)
        assert isinstance(op_repr, str)
        assert "MAngleEmbedding" in op_repr

    def test_ndim_params_property(self):
        op = MAngleEmbedding(np.asarray([0.1, 0.2]), [0, 1])
        assert op.ndim_params == (1,)

    def test_init_with_wrong_params(self):
        with pytest.raises(ValueError):
            MAngleEmbedding(np.asarray([0.1, 0.2, 0.3]), [0, 1])

    def test_init_with_padding(self):
        op = MAngleEmbedding(
            np.asarray(
                [
                    0.1,
                ]
            ),
            [0, 1],
        )
        assert qml.math.shape(op.parameters) == (1, 2)


class TestMAngleEmbeddings:
    @pytest.mark.parametrize(
        "initial_binary_string, rot, is_adjoint, seed",
        [
            (i_b_string, rot, adjoint, seed)
            for rot in ["X", "Y", "Z", "X,Y", "X,Z", "Y,Z", "X,Y,Z"]
            for i_b_string in ["00", "01", "10", "11"]
            for adjoint in [True, False]
            for seed in range(10)
        ],
    )
    def test_against_pennylane_rn_params(
        self,
        initial_binary_string,
        rot,
        is_adjoint,
        seed,
    ):
        rn_state = np.random.RandomState(seed=seed)
        rn_params = rn_state.uniform(0.0, np.pi / 2, size=(2,))
        cls_params_wires_list = [(MAngleEmbeddings, rn_params, [0, 1], {"rotations": rot})]
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

    def test_repr(self):
        op = MAngleEmbeddings(np.asarray([0.1, 0.2]), [0, 1])
        op_repr = repr(op)
        assert isinstance(op_repr, str)
        assert "MAngleEmbeddings" in op_repr

    def test_ndim_params_property(self):
        op = MAngleEmbeddings(np.asarray([0.1, 0.2]), [0, 1])
        assert op.ndim_params == (1,)
