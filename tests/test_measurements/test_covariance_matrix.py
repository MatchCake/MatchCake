import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.wires import Wires

import matchcake as mc
from matchcake.measurements import CovarianceMatrixMP, covariance_matrix
from matchcake.utils.covariance_matrix_toolbox import CovarianceMatrixToolbox

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
)

N_WIRES = 4


class TestCovarianceMatrix:
    @staticmethod
    def gates(angles):
        """Queue an entangling circuit on ``N_WIRES`` wires.

        :param angles: Three rotation angles.
        """
        qml.BasisState(np.array([1, 0, 1, 0]), wires=range(N_WIRES))
        qml.IsingXY(angles[0], wires=[0, 1])
        qml.IsingXY(angles[1], wires=[1, 2])
        qml.IsingXY(angles[2], wires=[2, 3])
        qml.IsingXX(angles[0] * angles[1], wires=[1, 2])

    @staticmethod
    def run(device_name: str, measurement, angles=(0.4, -0.8, 1.3)):
        """Execute the test circuit on a device.

        :param device_name: Name of the PennyLane device.
        :type device_name: str
        :param measurement: Callable that returns the measurement process.
        :param angles: Rotation angles of the circuit.
        :return: The result of the measurement.
        """

        @qml.qnode(qml.device(device_name, wires=N_WIRES))
        def circuit():
            TestCovarianceMatrix.gates(angles)
            return measurement()

        return circuit()

    @pytest.fixture
    def full_covariance(self) -> np.ndarray:
        """Covariance matrix of the final state of the test circuit, from its dense state vector."""
        state = np.asarray(self.run("default.qubit", qml.state))
        return np.asarray(CovarianceMatrixToolbox.from_density_matrix(np.outer(state, state.conj())))

    def test_is_exported(self):
        assert mc.covariance_matrix is covariance_matrix
        assert mc.measurements.CovarianceMatrixMP is CovarianceMatrixMP

    def test_measurement_process_attributes(self):
        measurement = covariance_matrix(wires=[1, 2])
        assert isinstance(measurement, CovarianceMatrixMP)
        assert measurement.wires == Wires([1, 2])
        assert measurement.gate_barrier is True
        assert measurement.numeric_type is float
        assert str(measurement) == "covariance_matrix"
        assert measurement.shape() == (4, 4)
        assert covariance_matrix().shape(num_device_wires=3) == (6, 6)
        assert covariance_matrix().wires == Wires([])

    def test_hash_depends_on_wires_and_gate_barrier(self):
        assert covariance_matrix(wires=[0, 1]).hash == covariance_matrix(wires=[0, 1]).hash
        assert covariance_matrix(wires=[0, 1]).hash != covariance_matrix(wires=[1, 2]).hash
        assert covariance_matrix(wires=[0, 2]).hash != covariance_matrix(wires=[0, 2], gate_barrier=False).hash

    def test_flatten_round_trip_keeps_gate_barrier(self):
        measurement = covariance_matrix(wires=[0, 2], gate_barrier=False)
        rebuilt = CovarianceMatrixMP._unflatten(*measurement._flatten())
        assert rebuilt.wires == measurement.wires
        assert rebuilt.gate_barrier is False

    def test_is_queued_in_a_tape(self):
        with qml.tape.QuantumTape() as tape:
            covariance_matrix(wires=[0, 1])
        assert isinstance(tape.measurements[0], CovarianceMatrixMP)

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed", "nif.qubit"])
    @pytest.mark.parametrize("wires", [None, [0], [0, 1], [1, 2, 3], [0, 1, 2, 3]])
    def test_contiguous_wires_match_dense_covariance(self, device_name, wires, full_covariance):
        result = self.run(device_name, lambda: covariance_matrix(wires=wires))
        positions = list(range(N_WIRES)) if wires is None else wires
        indices = np.stack([2 * np.asarray(positions), 2 * np.asarray(positions) + 1], axis=1).reshape(-1)
        np.testing.assert_allclose(
            qml.math.toarray(result),
            full_covariance[np.ix_(indices, indices)],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed", "nif.qubit"])
    @pytest.mark.parametrize("wires", [[0, 2], [1, 0], [0, 3]])
    def test_other_wires_need_to_give_up_the_gate_barrier(self, device_name, wires, full_covariance):
        with pytest.raises(ValueError, match="gate_barrier=False"):
            self.run(device_name, lambda: covariance_matrix(wires=wires))
        with pytest.warns(UserWarning, match="without gate barriers"):
            result = self.run(device_name, lambda: covariance_matrix(wires=wires, gate_barrier=False))
        indices = np.stack([2 * np.asarray(wires), 2 * np.asarray(wires) + 1], axis=1).reshape(-1)
        np.testing.assert_allclose(
            qml.math.toarray(result),
            full_covariance[np.ix_(indices, indices)],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_process_state_and_density_matrix_agree(self):
        state = np.asarray(self.run("default.qubit", qml.state))
        measurement = covariance_matrix(wires=[1, 2])
        np.testing.assert_allclose(
            measurement.process_state(state, Wires(range(N_WIRES))),
            measurement.process_density_matrix(np.outer(state, state.conj()), Wires(range(N_WIRES))),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("device_name", ["default.qubit", "nif.qubit"])
    def test_gradient(self, device_name):
        angles = torch.tensor([0.4, -0.8, 1.3], dtype=torch.float64, requires_grad=True)

        def reduced_covariance(raw_angles):
            return self.run(device_name, lambda: covariance_matrix(wires=[1, 2]), raw_angles)

        assert torch.autograd.gradcheck(
            reduced_covariance, (angles,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )
