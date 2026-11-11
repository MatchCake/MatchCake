import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError
from pennylane.measurements import ExpectationMP

from matchcake import NonInteractingFermionicDevice

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    TEST_SEED,
    set_seed,
)

N_SHOTS = 20_000

DIAGONAL_OBSERVABLES = [
    qml.Z(0),
    qml.Z(0) @ qml.Z(1),
    qml.Identity(0),
    2.0 * qml.Z(0) + qml.Z(1),
    qml.Hamiltonian([1.0, 2.0], [qml.Z(0), qml.Z(0) @ qml.Z(1)]),
    qml.Projector([1], wires=0),
    qml.Hermitian(np.diag([1.0, -1.0]), wires=0),
    qml.Hermitian(np.diag([0.5, -2.0]), wires=1),
    qml.Hermitian(np.diag([1.0, 2.0, 3.0, 4.0]), wires=[0, 1]),
    qml.Hermitian(np.diag([4.0, -1.0, 2.0, 0.0]), wires=[1, 0]),
]

# Sampled in a product basis of single-qubit Paulis.
PAULI_PRODUCT_BASIS_OBSERVABLES = [
    qml.X(0),
    qml.Y(1),
    qml.X(0) @ qml.X(1),
    qml.Z(0) @ qml.Y(1),
    qml.Z(0) + qml.X(1),
    qml.Hamiltonian([1.0, 2.0], [qml.Z(0), qml.X(0) @ qml.X(1)]),
]

NON_DIAGONAL_OBSERVABLES = [
    qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=0),
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
        analytic_device = qml.device("default.qubit", wires=2)
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        expected = qml.QNode(self.circuit, analytic_device)(observable)
        estimate = qml.QNode(self.circuit, shots_device)(observable)
        np.testing.assert_allclose(
            float(estimate), float(expected), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("observable", PAULI_PRODUCT_BASIS_OBSERVABLES)
    def test_pauli_product_basis_observable_expval_matches_analytic(self, observable):
        analytic_device = qml.device("default.qubit", wires=2)
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

    @pytest.mark.parametrize("observable", NON_DIAGONAL_OBSERVABLES)
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

    def test_diagonal_hermitian_sample_values_with_shots(self):
        observable = qml.Hermitian(np.diag([3.0, 5.0]), wires=0)
        analytic_device = qml.device("default.qubit", wires=2)
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        expected = qml.QNode(self.circuit, analytic_device)(observable)
        samples = np.asarray(qml.QNode(self.sample_circuit, shots_device)(observable))
        np.testing.assert_array_equal(np.unique(samples), np.array([3.0, 5.0]))
        np.testing.assert_allclose(
            np.mean(samples), float(expected), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize(
        "observable, expected",
        [
            (qml.Z(0) @ qml.Z(1), True),
            (2.0 * qml.Z(0) + qml.Z(1), True),
            (qml.Projector([1], wires=0), True),
            (qml.X(0), False),
            (qml.Hermitian(np.diag([1.0, -1.0]), wires=0), False),
        ],
    )
    def test_can_process_samples_directly(self, observable, expected):
        assert NonInteractingFermionicDevice._can_process_samples_directly(observable) is expected

    def test_can_process_samples_directly_without_diagonalizing_gates(self):
        observable = type("HermitianWithoutDiagonalizingGates", (qml.Hermitian,), {"has_diagonalizing_gates": False})(
            np.eye(2), wires=0
        )
        assert NonInteractingFermionicDevice._can_process_samples_directly(observable) is False

    @pytest.mark.parametrize(
        "observable, expected",
        [
            (qml.Hermitian(np.diag([1.0, -1.0]), wires=0), np.array([1.0, -1.0])),
            (qml.Hermitian(np.diag([1.0, 2.0, 3.0, 4.0]), wires=[0, 1]), np.array([1.0, 2.0, 3.0, 4.0])),
            (qml.Z(0), np.array([1.0, -1.0])),
        ],
    )
    def test_diagonal_matrix_eigvals(self, observable, expected):
        eigvals = NonInteractingFermionicDevice._diagonal_matrix_eigvals(observable)
        np.testing.assert_allclose(eigvals, expected, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.Hermitian(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=0),
            qml.X(0),
            type("BatchedHermitian", (qml.Hermitian,), {"batch_size": 2})(np.eye(2), wires=0),
            type("HermitianWithoutMatrix", (qml.Hermitian,), {"has_matrix": False})(np.eye(2), wires=0),
        ],
    )
    def test_diagonal_matrix_eigvals_returns_none(self, observable):
        assert NonInteractingFermionicDevice._diagonal_matrix_eigvals(observable) is None

    def test_build_sampled_measurement_accepts_none(self):
        shots_device = NonInteractingFermionicDevice(wires=2, shots=N_SHOTS)
        measurement = shots_device._build_sampled_measurement(ExpectationMP, None)
        assert isinstance(measurement, ExpectationMP)
        assert measurement.obs is None

    @staticmethod
    def batched_circuit(theta, measurement_fn):
        qml.BasisState(np.array([0, 1]), wires=[0, 1])
        qml.SingleExcitation(theta, wires=[0, 1])
        return measurement_fn()

    @pytest.mark.parametrize(
        "measurement_fn",
        [
            lambda: qml.expval(qml.Z(0)),
            lambda: qml.expval(qml.X(0) @ qml.X(1)),
            lambda: qml.probs(wires=[0]),
            lambda: (qml.expval(qml.Z(1)), qml.probs(wires=[0, 1])),
        ],
    )
    def test_batched_circuit_matches_default_qubit(self, measurement_fn):
        theta = np.array([0.1, 0.8, 2.0])
        expected = qml.set_shots(qml.QNode(self.batched_circuit, qml.device("default.qubit", wires=2)), None)(
            theta, measurement_fn
        )
        estimate = qml.QNode(self.batched_circuit, NonInteractingFermionicDevice(wires=2, shots=N_SHOTS))(
            theta, measurement_fn
        )
        if not isinstance(expected, tuple):
            expected, estimate = (expected,), (estimate,)
        for expected_result, estimated_result in zip(expected, estimate):
            np.testing.assert_allclose(
                np.asarray(estimated_result, dtype=float),
                np.asarray(expected_result, dtype=float),
                atol=ATOL_APPROX_COMPARISON,
                rtol=RTOL_APPROX_COMPARISON,
            )

    @pytest.mark.parametrize(
        "measurement_fn, expected_shape",
        [
            (lambda: qml.sample(), (3, N_SHOTS, 2)),
            (lambda: qml.sample(wires=[1]), (3, N_SHOTS, 1)),
            (lambda: qml.sample(qml.Z(0)), (3, N_SHOTS)),
            (lambda: qml.sample(qml.X(0) @ qml.X(1)), (3, N_SHOTS)),
        ],
    )
    def test_batched_circuit_sample_shape(self, measurement_fn, expected_shape):
        samples = qml.QNode(self.batched_circuit, NonInteractingFermionicDevice(wires=2, shots=N_SHOTS))(
            np.array([0.1, 0.8, 2.0]), measurement_fn
        )
        assert np.shape(samples) == expected_shape

    def test_samples_in_pennylane_layout_without_samples_raises(self):
        with pytest.raises(ValueError, match="No samples have been generated"):
            NonInteractingFermionicDevice(wires=2)._samples_in_pennylane_layout()
