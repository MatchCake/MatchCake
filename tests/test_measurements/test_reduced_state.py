import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.exceptions import DeviceError
from pennylane.wires import Wires

import matchcake as mc
from matchcake.measurements import ReducedStateMP, reduced_state
from matchcake.operations import CompRxRx, CompRyRy
from matchcake.operations.state_preparation import GaussianStatePrep, PlusState

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)

N_WIRES = 4


class TestReducedState:
    @staticmethod
    def gates(angles, is_displaced: bool = False):
        """Queue an entangling matchgate circuit on ``N_WIRES`` wires.

        :param angles: Two rotation angles, or a batch of them.
        :param is_displaced: Whether to start from a product state with superposed qubits.
        :type is_displaced: bool
        """
        if is_displaced:
            PlusState(wires=range(N_WIRES))
        else:
            qml.BasisState(np.array([1, 0, 1, 0]), wires=range(N_WIRES))
        CompRxRx(angles, wires=[0, 1])
        CompRyRy(angles, wires=[1, 2])
        CompRxRx(angles, wires=[2, 3])

    @classmethod
    def run(
        cls, measurement, angles=(0.4, -0.8), device_name: str = "nif.qubit", is_displaced: bool = False, shots=None
    ):
        """Execute the test circuit on a device.

        :param measurement: Callable that returns the measurement processes.
        :param angles: Rotation angles of the circuit.
        :param device_name: Name of the PennyLane device.
        :type device_name: str
        :param is_displaced: Whether to start from a product state with superposed qubits.
        :type is_displaced: bool
        :param shots: Number of shots, or None for an analytic execution.
        :return: The result of the measurement.
        """

        @qml.set_shots(shots)
        @qml.qnode(qml.device(device_name, wires=N_WIRES))
        def circuit():
            cls.gates(qml.math.asarray(angles), is_displaced)
            return measurement()

        return circuit()

    @classmethod
    def device_after_the_circuit(cls, angles=(0.4, -0.8), is_displaced: bool = False):
        """Apply the test circuit to a device, outside of a QNode.

        :param angles: Rotation angles of the circuit.
        :param is_displaced: Whether to start from a product state with superposed qubits.
        :type is_displaced: bool
        :return: The device.
        """
        with qml.queuing.AnnotatedQueue() as queue:
            cls.gates(np.asarray(angles), is_displaced)
        device = mc.NonInteractingFermionicDevice(wires=N_WIRES)
        device.apply(queue.queue)
        return device

    def test_is_exported(self):
        assert mc.reduced_state is reduced_state
        assert mc.measurements.ReducedStateMP is ReducedStateMP

    def test_measurement_process_attributes(self):
        measurement = reduced_state(wires=[1, 2])
        assert isinstance(measurement, ReducedStateMP)
        assert measurement.wires == Wires([1, 2])
        assert measurement.reduced_wires is None
        assert measurement.gate_barrier is True
        assert str(measurement) == "reduced_state"
        assert reduced_state().wires == Wires([])
        measurement = reduced_state(wires=[1, 2], reduced_wires=["a", "b"], gate_barrier=False)
        assert measurement.reduced_wires == Wires(["a", "b"])
        assert measurement.gate_barrier is False

    def test_hash_depends_on_the_arguments(self):
        assert reduced_state(wires=[0, 1]).hash == reduced_state(wires=[0, 1]).hash
        assert reduced_state(wires=[0, 1]).hash != reduced_state(wires=[1, 2]).hash
        assert reduced_state(wires=[0, 2]).hash != reduced_state(wires=[0, 2], gate_barrier=False).hash
        assert reduced_state(wires=[0, 2]).hash != reduced_state(wires=[0, 2], reduced_wires=[3, 4]).hash

    def test_flatten_round_trip(self):
        measurement = reduced_state(wires=[0, 2], reduced_wires=["a", "b"], gate_barrier=False)
        rebuilt = ReducedStateMP._unflatten(*measurement._flatten())
        assert rebuilt.wires == measurement.wires
        assert rebuilt.reduced_wires == Wires(["a", "b"])
        assert rebuilt.gate_barrier is False

    def test_is_queued_in_a_tape(self):
        with qml.tape.QuantumTape() as tape:
            reduced_state(wires=[0, 1])
        assert isinstance(tape.measurements[0], ReducedStateMP)

    @pytest.mark.parametrize("is_displaced", [False, True])
    @pytest.mark.parametrize("wires", [None, [0, 1], [1, 2, 3], [0, 2], [0, 3], [1, 3]])
    def test_returns_the_reduced_state_of_the_device(self, wires, is_displaced):
        result = self.run(lambda: reduced_state(wires=wires), is_displaced=is_displaced)
        expected = self.device_after_the_circuit(is_displaced=is_displaced).reduced_state(
            list(range(N_WIRES)) if wires is None else wires
        )
        assert isinstance(result, GaussianStatePrep)
        assert result.wires == expected.wires
        assert result.barriers == expected.barriers
        assert result.num_params == expected.num_params
        for value, expected_value in zip(result.data, expected.data):
            np.testing.assert_allclose(
                qml.math.toarray(value),
                qml.math.toarray(expected_value),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    def test_options_are_passed_to_the_device(self):
        result = self.run(lambda: reduced_state(wires=[0, 2], reduced_wires=["a", "b"]))
        assert result.wires == Wires(["a", "b"])
        assert result.barriers == (1,)
        with pytest.warns(UserWarning, match="without gate barriers"):
            result = self.run(lambda: reduced_state(wires=[0, 2], gate_barrier=False))
        assert result.barriers == ()

    def test_with_other_measurements(self):
        state, expval, probabilities = self.run(
            lambda: (reduced_state(wires=[0, 1]), qml.expval(qml.Z(0)), qml.probs(wires=[0, 1]))
        )
        assert isinstance(state, GaussianStatePrep)
        np.testing.assert_allclose(
            float(expval),
            float(self.run(lambda: qml.expval(qml.Z(0)))),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        assert qml.math.shape(probabilities) == (4,)

    def test_batched_circuit(self):
        angles = torch.tensor([[0.4, -0.8], [0.1, 0.2], [1.3, 0.5]], dtype=torch.float64)
        result = self.run(lambda: reduced_state(wires=[0, 1, 2]), angles=angles)
        assert tuple(result.covariance_matrix.shape) == (3, 6, 6)
        for index in range(3):
            single = self.run(lambda: reduced_state(wires=[0, 1, 2]), angles=angles[index])
            np.testing.assert_allclose(
                qml.math.toarray(result.covariance_matrix)[index],
                qml.math.toarray(single.covariance_matrix),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    def test_gradients_flow_through_the_returned_state(self):
        angles = torch.tensor([0.4, -0.8], dtype=torch.float64, requires_grad=True)

        @qml.qnode(qml.device("nif.qubit", wires=N_WIRES), interface="torch")
        def circuit(parameters):
            self.gates(parameters, is_displaced=True)
            return reduced_state(wires=[0, 1])

        result = circuit(angles)
        assert result.covariance_matrix.requires_grad
        assert result.displacement_vector.requires_grad
        (gradient,) = torch.autograd.grad(result.covariance_matrix.sum() + result.displacement_vector.sum(), angles)
        assert torch.all(torch.isfinite(gradient))
        assert gradient.abs().max() > 0

    def test_is_rejected_with_shots(self):
        with pytest.raises(DeviceError, match="analytic"):
            self.run(lambda: reduced_state(wires=[0, 1]), shots=10)

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_is_only_supported_by_the_nif_device(self, device_name):
        with pytest.raises(NotImplementedError, match="only supported by nif.qubit"):
            self.run(lambda: reduced_state(wires=[0, 1]), device_name=device_name)

    def test_process_methods_are_not_implemented(self):
        measurement = reduced_state(wires=[0, 1])
        with pytest.raises(NotImplementedError, match="only supported by nif.qubit"):
            measurement.process_state(np.ones(4) / 2, Wires([0, 1]))
        with pytest.raises(NotImplementedError, match="only supported by nif.qubit"):
            measurement.process_density_matrix(np.eye(4) / 4, Wires([0, 1]))
