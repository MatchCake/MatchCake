import numpy as np
import pennylane as qml
import pytest
from pennylane import X

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.probability_strategies import ProductStateProbabilityStrategy
from matchcake.operations import CompRyRy
from matchcake.operations.state_preparation.minus_state import MinusState
from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)


class TestMinusState:
    @staticmethod
    def matchgate_angles(num_wires):
        """Deterministic ``CompRyRy`` angles for a nearest-neighbour brick-wall layer.

        ``CompRyRy`` is used rather than ``CompRxRx`` because it maps the uniform
        distribution of ``|-...->`` onto a non-uniform one, so the comparison against
        ``default.qubit`` is sensitive to an incorrect probability path.

        :param num_wires: Number of wires.
        :type num_wires: int
        :return: Angles of shape ``(num_wires - 1, 2)``.
        :rtype: np.ndarray
        """
        return np.random.default_rng(TEST_SEED).normal(size=(num_wires - 1, 2))

    @staticmethod
    def selected_strategy(device):
        """Return the strategy the device's dispatcher routes the current preparation to.

        :param device: Device whose ``prob_dispatcher`` chain is inspected.
        :type device: NonInteractingFermionicDevice
        :return: The selected probability strategy.
        :rtype: matchcake.devices.probability_strategies.ProbabilityStrategy
        """
        return next(s for s in device.prob_dispatcher.strategies if s.can_execute(device.state_prep_op))

    @pytest.mark.parametrize("num_wires", [2, 3, 4, 5])
    def test_state_vector(self, num_wires):
        qubit_dev = qml.device("lightning.qubit", wires=num_wires)

        def circuit():
            MinusState(wires=qubit_dev.wires)
            return qml.state()

        qubit_qnode = qml.QNode(circuit, qubit_dev)
        state = qubit_qnode()
        odd_binary_states = np.asarray([bin(i).count("1") % 2 for i in range(2**num_wires)])
        target_state = np.ones(2**num_wires) / np.sqrt(2**num_wires) * (-1) ** odd_binary_states
        np.testing.assert_allclose(state, target_state, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_vs_svs(self):
        pauli_string = [X(0) @ X(1)]
        nif_dev = NonInteractingFermionicDevice(wires=2)
        qubit_dev = qml.device("lightning.qubit", wires=2)

        def circuit():
            MinusState(wires=nif_dev.wires)
            return qml.expval(sum(pauli_string))

        nif_qnode = qml.QNode(circuit, nif_dev)
        qubit_qnode = qml.QNode(circuit, qubit_dev)
        expected_value = qubit_qnode()
        actual_value = nif_qnode()

        np.testing.assert_allclose(
            actual_value,
            expected_value,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("num_wires", [2, 3, 4])
    def test_probs_against_qubit_device(self, num_wires):
        nif_dev = NonInteractingFermionicDevice(wires=num_wires)
        qubit_dev = qml.device("default.qubit", wires=num_wires)

        def circuit():
            MinusState(wires=range(num_wires))
            return qml.probs(wires=range(num_wires))

        actual_value = qml.math.detach(qml.QNode(circuit, nif_dev)())
        expected_value = qml.math.detach(qml.QNode(circuit, qubit_dev)())

        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("num_wires", [2, 3, 4])
    def test_probs_with_matchgates_against_qubit_device(self, num_wires):
        nif_dev = NonInteractingFermionicDevice(wires=num_wires)
        qubit_dev = qml.device("default.qubit", wires=num_wires)
        angles = self.matchgate_angles(num_wires)

        def circuit():
            MinusState(wires=range(num_wires))
            for wire in range(num_wires - 1):
                CompRyRy(angles[wire], wires=[wire, wire + 1])
            return qml.probs(wires=range(num_wires))

        actual_value = np.asarray(qml.math.detach(qml.QNode(circuit, nif_dev)()))
        expected_value = np.asarray(qml.math.detach(qml.QNode(circuit, qubit_dev)()))

        # Guard against a vacuous comparison: the layer must break the uniformity of |-...->.
        assert expected_value.max() - expected_value.min() > ATOL_APPROX_COMPARISON
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_probs_routes_to_product_state_strategy(self):
        # |-> is not a computational-basis state, so the single-basis-state strategies must
        # decline it and let it fall through to the covariance-matrix strategy.
        nif_dev = NonInteractingFermionicDevice(wires=2)
        nif_dev.apply([MinusState(wires=[0, 1])])
        assert isinstance(self.selected_strategy(nif_dev), ProductStateProbabilityStrategy)

    def test_probs_with_matchgates_routes_to_product_state_strategy(self):
        nif_dev = NonInteractingFermionicDevice(wires=2)
        angles = self.matchgate_angles(2)
        nif_dev.apply([MinusState(wires=[0, 1]), CompRyRy(angles[0], wires=[0, 1])])
        assert isinstance(self.selected_strategy(nif_dev), ProductStateProbabilityStrategy)

    def test_label(self):
        op = MinusState(wires=[0, 1])
        assert "-" in op.label()
