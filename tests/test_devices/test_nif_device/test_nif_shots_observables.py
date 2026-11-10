import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError

from matchcake import NonInteractingFermionicDevice

from ...configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON, TEST_SEED, set_seed

N_SHOTS = 20_000

DIAGONAL_OBSERVABLES = [
    qml.Z(0),
    qml.Z(0) @ qml.Z(1),
    qml.Identity(0),
    2.0 * qml.Z(0) + qml.Z(1),
    qml.Hamiltonian([1.0, 2.0], [qml.Z(0), qml.Z(0) @ qml.Z(1)]),
    qml.Projector([1], wires=0),
]

NON_DIAGONAL_OBSERVABLES = [
    qml.X(0),
    qml.Y(1),
    qml.X(0) @ qml.X(1),
    qml.Z(0) @ qml.Y(1),
    qml.Z(0) + qml.X(1),
    qml.Hamiltonian([1.0, 2.0], [qml.Z(0), qml.X(0) @ qml.X(1)]),
    qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=0),
]

# PennyLane rejects ``qml.sample`` of sums before they reach the device.
NON_DIAGONAL_SAMPLE_OBSERVABLES = [
    observable for observable in NON_DIAGONAL_OBSERVABLES if not isinstance(observable, qml.ops.Sum)
]


class TestNonInteractingFermionicDeviceShotsObservables:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @staticmethod
    def circuit(observable: qml.operation.Operator):
        qml.BasisState(np.array([0, 1]), wires=[0, 1])
        qml.SingleExcitation(0.8, wires=[0, 1])
        return qml.expval(observable)

    @staticmethod
    def sample_circuit(observable: qml.operation.Operator):
        qml.BasisState(np.array([0, 1]), wires=[0, 1])
        qml.SingleExcitation(0.8, wires=[0, 1])
        return qml.sample(observable)

    @pytest.mark.parametrize("observable", DIAGONAL_OBSERVABLES)
    def test_diagonal_observable_expval_matches_analytic(self, observable):
        analytic_device = NonInteractingFermionicDevice(wires=2)
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        expected = qml.QNode(self.circuit, analytic_device)(observable)
        estimate = qml.QNode(self.circuit, shots_device)(observable)
        np.testing.assert_allclose(
            float(estimate), float(expected), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("observable", NON_DIAGONAL_OBSERVABLES)
    def test_non_diagonal_observable_expval_raises_with_shots(self, observable):
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        with pytest.raises(DeviceError, match="not diagonal in the computational basis"):
            qml.QNode(self.circuit, shots_device)(observable)

    @pytest.mark.parametrize("observable", NON_DIAGONAL_SAMPLE_OBSERVABLES)
    def test_non_diagonal_observable_sample_raises_with_shots(self, observable):
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        with pytest.raises(DeviceError, match="not diagonal in the computational basis"):
            qml.QNode(self.sample_circuit, shots_device)(observable)

    def test_diagonal_observable_sample_with_shots(self):
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        samples = qml.QNode(self.sample_circuit, shots_device)(qml.Z(0) @ qml.Z(1))
        np.testing.assert_array_equal(np.unique(np.asarray(samples)), np.array([-1]))

    def test_non_diagonal_observable_analytic_expval_is_supported(self):
        analytic_device = NonInteractingFermionicDevice(wires=2)
        expected = qml.QNode(self.circuit, qml.device("default.qubit", wires=2))(qml.X(0) @ qml.X(1))
        result = qml.QNode(self.circuit, analytic_device)(qml.X(0) @ qml.X(1))
        np.testing.assert_allclose(
            float(result), float(expected), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_execute_generator_non_diagonal_observable_raises_with_shots(self):
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        operations = [qml.BasisState(np.array([0, 1]), wires=[0, 1]), qml.SingleExcitation(0.8, wires=[0, 1])]
        with pytest.raises(DeviceError, match="not diagonal in the computational basis"):
            shots_device.execute_generator(iter(operations), observable=qml.X(0) @ qml.X(1), output_type="expval")

    @pytest.mark.parametrize(
        "observable, expected",
        [(observable, True) for observable in DIAGONAL_OBSERVABLES]
        + [(observable, False) for observable in NON_DIAGONAL_OBSERVABLES],
    )
    def test_is_computational_basis_diagonal(self, observable, expected):
        assert NonInteractingFermionicDevice._is_computational_basis_diagonal(observable) is expected

    def test_is_computational_basis_diagonal_without_diagonalizing_gates(self):
        observable = qml.Hermitian(np.eye(2), wires=0)
        observable_without_gates = type(
            "HermitianWithoutDiagonalizingGates", (qml.Hermitian,), {"has_diagonalizing_gates": False}
        )(np.eye(2), wires=0)
        assert NonInteractingFermionicDevice._is_computational_basis_diagonal(observable) is False
        assert NonInteractingFermionicDevice._is_computational_basis_diagonal(observable_without_gates) is False

    def test_check_observable_can_be_sampled_accepts_none(self):
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        assert shots_device._check_observable_can_be_sampled(None) is None
