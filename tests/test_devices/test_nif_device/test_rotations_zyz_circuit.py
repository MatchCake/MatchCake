import numpy as np
import pennylane as qml
import pytest

from matchcake import operations, utils
from matchcake.utils.torch_utils import to_numpy

from ...configs import (
    ATOL_APPROX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
)
from .. import devices_init


@pytest.mark.parametrize(
    "theta,contraction_strategy",
    [
        (theta, contraction_strategy)
        for theta in np.linspace(0, 2 * np.pi, num=N_RANDOM_TESTS_PER_CASE)
        for contraction_strategy in [
            None,
            "neighbours",
            "forward",
            "horizontal",
            "vertical",
        ]
    ],
)
def test_multiples_matchgate_state_with_qubit_device_zyz(theta, contraction_strategy):
    initial_binary_string = "00"
    initial_binary_state = utils.binary_string_to_vector(initial_binary_string)
    nif_device, qubit_device = devices_init(wires=len(initial_binary_state), contraction_strategy=contraction_strategy)

    def circuit_state():
        operations.CompRzRz(np.asarray([theta, theta]), wires=[0, 1])
        operations.CompRyRy(np.asarray([theta, theta]), wires=[0, 1])
        operations.CompRzRz(np.asarray([theta, theta]), wires=[0, 1])
        return qml.state()

    def circuit_probs():
        operations.CompRzRz(np.asarray([theta, theta]), wires=[0, 1])
        operations.CompRyRy(np.asarray([theta, theta]), wires=[0, 1])
        operations.CompRzRz(np.asarray([theta, theta]), wires=[0, 1])
        return qml.probs()

    qubit_state = qml.QNode(circuit_state, qubit_device)()
    expected_state = np.asarray([np.exp(-1j * theta) * np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

    np.testing.assert_allclose(
        qubit_state.squeeze(),
        expected_state.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )

    qubit_probs = qml.QNode(circuit_probs, qubit_device)()
    nif_probs = qml.QNode(circuit_probs, nif_device)()
    qubit_probs = to_numpy(qubit_probs).squeeze()
    nif_probs = to_numpy(nif_probs).squeeze()

    np.testing.assert_allclose(
        nif_probs.squeeze(),
        qubit_probs.squeeze(),
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
