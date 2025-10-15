import numpy as np
import pennylane as qml
import pytest

from matchcake import Matchgate, MatchgateOperation, NonInteractingFermionicDevice
from matchcake import matchgate_parameter_sets as mps
from matchcake import utils
from matchcake.utils import pfaffian
from matchcake.utils.math import dagger, det

from .configs import (
    ATOL_MATRIX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)
from .test_nif_device import single_matchgate_circuit

set_seed(TEST_SEED)

fSWAP_R = Matchgate(mps.fSWAP).single_particle_transition_matrix


@pytest.mark.parametrize(
    "params,initial_binary_state,output_state",
    [
        (mps.Identity, "00", np.array([1, 0, 0, 0])),
        (mps.Identity, "01", np.array([0, 1, 0, 0])),
        (mps.Identity, "10", np.array([0, 0, 1, 0])),
        (mps.Identity, "11", np.array([0, 0, 0, 1])),
        (mps.fSWAP, "00", np.array([1, 0, 0, 0])),
        (mps.fSWAP, "01", np.array([0, 0, 1, 0])),
        (mps.fSWAP, "10", np.array([0, 1, 0, 0])),
        (mps.fSWAP, "11", np.array([0, 0, 0, -1])),
        (mps.HellParams, "00", np.array([-0.5, 0, 0, np.sqrt(0.75) * 1j])),
    ],
)
def test_single_matchgate_circuit_output_state(params, initial_binary_state, output_state):
    from .test_nif_device import devices_init

    _, qubit_device = devices_init()
    initial_state = utils.binary_string_to_vector(initial_binary_state)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    qubit_state = qubit_qnode(
        mps.MatchgatePolarParams.parse_from_any(params).to_numpy(),
        initial_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    np.testing.assert_allclose(
        qubit_state.squeeze(),
        output_state.squeeze(),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params,initial_binary_state,output_probs",
    [
        (mps.Identity, "00", np.array([1, 0, 0, 0])),
        (mps.Identity, "01", np.array([0, 1, 0, 0])),
        (mps.Identity, "10", np.array([0, 0, 1, 0])),
        (mps.Identity, "11", np.array([0, 0, 0, 1])),
        (mps.fSWAP, "00", np.array([1, 0, 0, 0])),
        (mps.fSWAP, "01", np.array([0, 0, 1, 0])),
        (mps.fSWAP, "10", np.array([0, 1, 0, 0])),
        (mps.fSWAP, "11", np.array([0, 0, 0, 1])),
        (mps.HellParams, "00", np.array([0.25, 0, 0, 0.75])),
    ],
)
def test_single_matchgate_circuit_output_probs(params, initial_binary_state, output_probs):
    from .test_nif_device import devices_init

    _, qubit_device = devices_init()
    initial_state = utils.binary_string_to_vector(initial_binary_state)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)
    qubit_state = qubit_qnode(
        mps.MatchgatePolarParams.parse_from_any(params).to_numpy(),
        initial_state,
        in_param_type=mps.MatchgatePolarParams,
        out_op="state",
    )
    qubit_probs = utils.get_probabilities_from_state(qubit_state)
    np.testing.assert_allclose(
        qubit_probs.squeeze(),
        output_probs.squeeze(),
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params,k,binary_state,observable",
    [
        (
            mps.fSWAP,
            0,
            "01",
            np.array([[0, 1, 0, 1], [-1, 0, 0, 0], [0, 0, 0, 1], [-1, 0, -1, 0]]),
        ),
    ],
)
def test_single_matchgate_obs_on_specific_cases(params, k, binary_state, observable):
    nif_device = NonInteractingFermionicDevice(wires=2)
    nif_device.apply(MatchgateOperation(params, wires=[0, 1]))
    state = utils.binary_state_to_state(binary_state)
    pred_obs = nif_device.lookup_table.get_observable(k, state)
    pred_pf = pfaffian(pred_obs)
    pf = pfaffian(observable)
    np.testing.assert_allclose(
        pred_obs,
        observable,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
    np.testing.assert_almost_equal(pred_pf, pf)


@pytest.mark.parametrize(
    "wires, n_wires, padded_matrix",
    [
        (
            [0, 1],
            2,
            fSWAP_R,
        ),
    ],
)
def test_get_padded_single_transition_particle_matrix(wires, n_wires, padded_matrix):
    mgo = MatchgateOperation(mps.fSWAP, wires=wires)
    all_wires = list(range(n_wires))
    pred_padded_matrix = mgo.get_padded_single_particle_transition_matrix(wires=all_wires)
    np.testing.assert_allclose(
        pred_padded_matrix,
        padded_matrix,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            mps.MatchgatePolarParams(r0=1, r1=1),
            np.array(
                [
                    [0.25 * np.trace(utils.get_majorana(i, 2) @ utils.get_majorana(j, 2)) for j in range(4)]
                    for i in range(4)
                ]
            ),
        ),
        (
            mps.MatchgatePolarParams(r0=1, r1=1),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            mps.fSWAP,
            np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ]
            ),
        ),
    ],
)
def test_single_transition_matrix(params, expected):
    expected = qml.math.array(expected)
    mgo = MatchgateOperation(params, wires=[0, 1])
    np.testing.assert_allclose(
        mgo.single_particle_transition_matrix.squeeze(),
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "op, op_adjoint",
    [
        (
            MatchgateOperation(mps.fSWAP, wires=[0, 1]),
            MatchgateOperation(mps.fSWAP, wires=[0, 1]),
        ),
        (
            MatchgateOperation(mps.Identity, wires=[0, 1]),
            MatchgateOperation(mps.Identity, wires=[0, 1]),
        ),
        (
            MatchgateOperation(mps.fHH, wires=[0, 1]),
            MatchgateOperation(mps.fHH, wires=[0, 1]),
        ),
        (
            MatchgateOperation(mps.HellParams, wires=[0, 1]),
            MatchgateOperation.from_matrix(
                qml.math.conj(qml.math.transpose(MatchgateOperation(mps.HellParams, wires=[0, 1]).matrix().squeeze())),
                wires=[0, 1],
            ),
        ),
    ],
)
def test_matchgate_operation_adjoint(op, op_adjoint):
    pred = op.adjoint().matrix().squeeze()
    expected = op_adjoint.matrix().squeeze()
    np.testing.assert_allclose(
        pred,
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "op0,op1,expected",
    [
        # (
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]),
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]),
        #         np.array(
        #             [
        #                 [1, 0, 0, 0],
        #                 [0, 1, 0, 0],
        #                 [0, 0, 1, 0],
        #                 [0, 0, 0, 1],
        #             ]
        #         )
        # ),
        # (
        #         MatchgateOperation(mps.Identity, wires=[0, 1]),
        #         MatchgateOperation(mps.Identity, wires=[0, 1]),
        #         np.array(
        #             [
        #                 [1, 0, 0, 0],
        #                 [0, 1, 0, 0],
        #                 [0, 0, 1, 0],
        #                 [0, 0, 0, 1],
        #             ]
        #         )
        # ),
        # (
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]),
        #         MatchgateOperation(mps.Identity, wires=[0, 1]),
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]).matrix().squeeze(),
        # ),
        # (
        #         MatchgateOperation(mps.Identity, wires=[0, 1]),
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]),
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]).matrix().squeeze(),
        # ),
        # (
        #         MatchgateOperation(mps.fHH, wires=[0, 1]),
        #         MatchgateOperation(mps.fHH, wires=[0, 1]),
        #         MatchgateOperation(mps.Identity, wires=[0, 1]).matrix().squeeze(),
        # ),
        # (
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]),
        #         MatchgateOperation(mps.fSWAP, wires=[0, 1]).adjoint(),
        #         MatchgateOperation(mps.Identity, wires=[0, 1]).matrix().squeeze(),
        # ),
        # (
        #         MatchgateOperation(mps.fHH, wires=[0, 1]),
        #         MatchgateOperation(mps.fHH, wires=[0, 1]).adjoint(),
        #         MatchgateOperation(mps.Identity, wires=[0, 1]).matrix().squeeze(),
        # ),
        (
            MatchgateOperation(mps.HellParams, wires=[0, 1]),
            MatchgateOperation(mps.HellParams, wires=[0, 1]).adjoint(),
            MatchgateOperation(mps.Identity, wires=[0, 1]).matrix().squeeze(),
        ),
    ],
)
def test_matchgate_operation_matmul_specific_cases(op0, op1, expected):
    pred = (op0 @ op1).matrix().squeeze()
    np.testing.assert_allclose(
        pred,
        expected,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "matchgate_params",
    [mps.MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_matchgate_sptm_unitary(matchgate_params):
    mgo = MatchgateOperation(matchgate_params, wires=[0, 1])
    sptm = mgo.single_particle_transition_matrix
    expected_eye = np.einsum("...ij,...jk->...ik", sptm, dagger(sptm))
    eye = np.zeros_like(expected_eye)
    eye[..., np.arange(expected_eye.shape[-1]), np.arange(expected_eye.shape[-1])] = 1
    np.testing.assert_allclose(
        expected_eye,
        eye,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )


@pytest.mark.parametrize(
    "matchgate_params",
    [mps.MatchgatePolarParams.random() for _ in range(N_RANDOM_TESTS_PER_CASE)],
)
def test_matchgate_sptm_det(matchgate_params):
    mgo = MatchgateOperation(matchgate_params, wires=[0, 1])
    sptm = mgo.single_particle_transition_matrix
    sptm_det = det(sptm)
    np.testing.assert_allclose(
        sptm_det,
        1.0,
        atol=ATOL_MATRIX_COMPARISON,
        rtol=RTOL_MATRIX_COMPARISON,
    )
