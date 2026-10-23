from typing import Callable

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane import numpy as pnp

from matchcake import BatchHamiltonian, NIFDevice
from matchcake.operations.state_preparation import ProductState

from ...configs import ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON

# Separates double precision (error ~1e-16) from a coefficient rounded through float32 (error ~1e-8
# on an O(1) value), which PennyLane applies to a python float multiplied into a torch result.
DOUBLE_PRECISION_ATOL = 1e-12

N_QUBITS = 3

# Words of the observables below, chosen with nonzero expectations on the circuit of this suite so
# that a stale or doubled coefficient changes the value.
WORDS = [qml.Z(0) @ qml.Z(1), qml.X(1) @ qml.X(2)]


class TestLiveObservableCoefficients:
    @staticmethod
    def circuit(angle: float = 0.4) -> None:
        """A matchgate circuit with nonzero expectations on :data:`WORDS` and on ``Z_0 Z_2``.

        :param angle: Angle of the first rotation, the parameter differentiated below.
        """
        qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))
        qml.IsingXX(angle, wires=[0, 1])
        qml.IsingYY(0.3, wires=[1, 2])

    @staticmethod
    def make_weights(interface: str, values: list):
        """A fresh weight array of the given interface.

        :param interface: ``"torch"``, ``"numpy"`` or ``"autograd"``.
        :param values: The weights.
        :return: A float64 torch tensor, numpy array or pennylane.numpy array.
        """
        if interface == "torch":
            return torch.tensor(values, dtype=torch.float64)
        return np.array(values) if interface == "numpy" else pnp.array(values)

    @classmethod
    def expectation(cls, device: qml.devices.Device, observable: qml.operation.Operator, angle=0.4):
        """``<observable>`` after :meth:`circuit`, through a torch QNode on ``device``.

        :param device: The device.
        :param observable: The observable.
        :param angle: Angle of the first rotation.
        :return: The expectation value.
        """

        @qml.qnode(device, interface="torch")
        def run():
            cls.circuit(angle)
            return qml.expval(observable)

        return run()

    @classmethod
    def reference(cls, observable: qml.operation.Operator) -> float:
        """The same expectation on ``default.qubit``, for an observable built fresh by the caller.

        :param observable: The observable.
        :return: The expectation value.
        :rtype: float
        """
        return float(cls.expectation(qml.device("default.qubit", wires=N_QUBITS), observable))

    def test_a_reused_sum_gives_the_same_value_every_call_and_is_left_untouched(self) -> None:
        """A shape-``(1,)`` weight makes PennyLane's ``Sum.terms()`` fall back to simplification,
        which accumulated into this very tensor on every QNode call."""
        weight = torch.tensor([0.5], dtype=torch.float64)
        observable = weight * qml.prod(qml.Z(0), qml.Z(2)) + 0.25 * qml.prod(qml.Z(0), qml.Z(2))
        expected = self.reference(0.75 * qml.prod(qml.Z(0), qml.Z(2)))
        device = NIFDevice(wires=N_QUBITS)
        for _ in range(3):
            np.testing.assert_allclose(
                float(self.expectation(device, observable)),
                expected,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
        np.testing.assert_allclose(weight.numpy(), [0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_direct_evaluation_gives_the_same_value_every_call_and_leaves_the_weight(self) -> None:
        """The same through ``exact_expval``, which reads the observable in the expval strategies."""
        device = NIFDevice(wires=N_QUBITS)
        self.expectation(device, qml.Z(0))
        weight = torch.tensor([0.5], dtype=torch.float64)
        observable = weight * qml.prod(qml.Z(0), qml.Z(2)) + 0.25 * qml.prod(qml.Z(0), qml.Z(2))
        expected = float(device.exact_expval(qml.Hamiltonian([0.75], [qml.prod(qml.Z(0), qml.Z(2))])))
        for _ in range(3):
            np.testing.assert_allclose(
                float(device.exact_expval(observable)),
                expected,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
        np.testing.assert_allclose(weight.numpy(), [0.5], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "interface, build",
        [
            ("torch", lambda weights: qml.dot(weights, WORDS)),
            ("torch", lambda weights: weights[0] * WORDS[0] + weights[1] * WORDS[1]),
            ("numpy", lambda weights: qml.Hamiltonian(weights, WORDS)),
            ("autograd", lambda weights: qml.Hamiltonian(weights, WORDS)),
            ("torch", lambda weights: qml.Hamiltonian([1.0], [qml.X(0)]) - qml.Hamiltonian(weights, WORDS)),
        ],
    )
    def test_an_in_place_update_is_seen_by_the_next_call(self, interface: str, build: Callable) -> None:
        """An optimizer updates its weight tensor in place; the observable built from it must follow."""
        weights = self.make_weights(interface, [0.5, 0.3])
        observable = build(weights)
        device = NIFDevice(wires=N_QUBITS)
        self.expectation(device, observable)
        with torch.no_grad():
            weights[0] = 5.0
            weights[1] = -1.7
        np.testing.assert_allclose(
            float(self.expectation(device, observable)),
            # Built from python floats: default.qubit cannot evaluate a nested Hamiltonian of tensors.
            self.reference(build([5.0, -1.7])),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_wrapped_observable_is_correct_after_its_hamiltonian_was_reused_in_a_sum(self) -> None:
        """``H + H`` adds into the shared coefficient tensors of ``H``'s cached ``pauli_rep``."""
        hamiltonian = qml.Hamiltonian(torch.tensor([0.5, 0.3], dtype=torch.float64), WORDS)
        observable = qml.adjoint(hamiltonian)
        _ = hamiltonian + hamiltonian
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), observable)),
            self.reference(qml.Hamiltonian([0.5, 0.3], WORDS)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "build",
        [
            lambda weights: qml.Hamiltonian(weights, WORDS),
            lambda weights: qml.dot(weights, WORDS),
            lambda weights: weights[0] * WORDS[0] + weights[1] * WORDS[1],
        ],
    )
    def test_the_coefficient_gradients_match_default_qubit(self, build: Callable) -> None:
        """The rebuilt observable's coefficients are computed from the caller's, so gradients flow."""
        gradients = []
        for device in (NIFDevice(wires=N_QUBITS), qml.device("default.qubit", wires=N_QUBITS)):
            weights = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)
            self.expectation(device, build(weights)).backward()
            gradients.append(weights.grad.numpy())
        np.testing.assert_allclose(gradients[0], gradients[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_the_circuit_gradient_matches_default_qubit_through_a_rebuilt_observable(self) -> None:
        observable = 0.5 * qml.Z(0) + 0.3 * qml.prod(qml.Z(1), qml.Z(2))
        gradients = []
        for device in (NIFDevice(wires=N_QUBITS), qml.device("default.qubit", wires=N_QUBITS)):
            angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
            self.expectation(device, observable, angle).backward()
            gradients.append(float(angle.grad))
        assert abs(gradients[1]) > ATOL_SCALAR_COMPARISON
        np.testing.assert_allclose(gradients[0], gradients[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "build",
        [
            lambda weight: 0.5 * qml.Z(0) + weight * qml.Z(1),
            lambda weight: qml.Z(0) + weight * qml.Z(1),
            lambda weight: qml.dot([0.5, weight], [qml.Z(0), qml.Z(1)]),
        ],
    )
    def test_a_constant_next_to_a_trainable_weight_is_evaluated_and_differentiated(self, build: Callable) -> None:
        """PennyLane's ``Hamiltonian`` rejects a list mixing a python float with a trainable tensor."""
        results = []
        for device in (NIFDevice(wires=N_QUBITS), qml.device("default.qubit", wires=N_QUBITS)):
            weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
            value = self.expectation(device, build(weight))
            value.backward()
            results.append((float(value), float(weight.grad)))
        np.testing.assert_allclose(results[0], results[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "build",
        [
            lambda weight: qml.Z(1) + weight * qml.Identity(0),
            lambda weight: qml.pow(qml.Z(0), 2) + weight * qml.Z(1),
            lambda weight: weight * qml.Z(0) + 0.25 * qml.Z(1),
        ],
    )
    def test_a_one_element_weight_next_to_scalars_is_evaluated(self, build: Callable) -> None:
        weight = torch.tensor([0.5], dtype=torch.float64)
        np.testing.assert_allclose(
            np.ravel(self.expectation(NIFDevice(wires=N_QUBITS), build(weight)).numpy()),
            np.ravel(self.reference(build(torch.tensor([0.5], dtype=torch.float64)))),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_python_coefficients_keep_double_precision(self) -> None:
        """Coefficients that are not exact in float32 expose a single-precision round trip."""
        observable = 1.234 * qml.Z(0) + 0.777 * qml.Z(1)
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), observable)),
            self.reference(1.234 * qml.Z(0) + 0.777 * qml.Z(1)),
            atol=DOUBLE_PRECISION_ATOL,
            rtol=0.0,
        )

    def test_a_real_observable_built_through_phases_stays_real_and_differentiable(self) -> None:
        """``X_0 Y_0 X_0 Y_0 = -I`` multiplies through two ``i`` phases; the expectation must come back
        as float64 rather than complex, or ``backward`` refuses it."""
        observable = qml.Hamiltonian([0.5, 0.1], [qml.X(0) @ qml.Y(0) @ qml.X(0) @ qml.Y(0) @ qml.Z(1), qml.Z(1)])
        angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        value = self.expectation(NIFDevice(wires=N_QUBITS), observable, angle)
        assert value.dtype == torch.float64
        value.backward()
        reference_angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        reference = self.expectation(
            qml.device("default.qubit", wires=N_QUBITS),
            qml.Hamiltonian([-0.5, 0.1], [qml.Z(1), qml.Z(1)]),
            reference_angle,
        )
        torch.real(reference).backward()
        np.testing.assert_allclose(
            (float(value), float(angle.grad)),
            (float(torch.real(reference)), float(reference_angle.grad)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_multiple_of_the_identity_is_evaluated(self) -> None:
        observable = qml.Hamiltonian(torch.tensor([0.7], dtype=torch.float64), [0.5 * qml.Identity(0)])
        np.testing.assert_allclose(
            np.ravel(self.expectation(NIFDevice(wires=N_QUBITS), observable).numpy()),
            [0.35],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_a_grouped_hamiltonian_neither_drifts_nor_mutates_its_weights(self, requires_grad: bool) -> None:
        """PennyLane's splitter accumulates the repeated words of a grouped Hamiltonian into its
        coefficient tensor in place, which drifts and fails outright on a trainable weight."""
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=requires_grad)
        observable = qml.Hamiltonian(weights, [qml.Z(0), qml.Z(0)], grouping_type="qwc")
        expected = self.reference(0.8 * qml.Z(0))
        device = NIFDevice(wires=N_QUBITS)
        for _ in range(3):
            np.testing.assert_allclose(
                float(self.expectation(device, observable)),
                expected,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
        np.testing.assert_allclose(
            weights.detach().numpy(), [0.5, 0.3], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_sum_with_a_non_pauli_term_does_not_drift(self) -> None:
        """The projector is carried whole, and the Pauli part is still read live."""
        weight = torch.tensor(0.5, dtype=torch.float64)
        observable = qml.Projector([1], wires=[0]) + weight * qml.Z(1) + 0.25 * qml.Z(1)
        expected = self.reference(qml.Projector([1], wires=[0]) + 0.75 * qml.Z(1))
        device = NIFDevice(wires=N_QUBITS)
        for _ in range(3):
            np.testing.assert_allclose(
                float(self.expectation(device, observable)),
                expected,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )

    @pytest.mark.parametrize("make_weight", [lambda: np.array([0.5]), lambda: pnp.array([0.5])])
    def test_a_one_element_numpy_weight_is_read_by_the_m_pfaffian_strategy(self, make_weight: Callable) -> None:
        device = NIFDevice(wires=N_QUBITS)
        self.expectation(device, qml.Z(0))
        observable = make_weight() * qml.Z(0) + 0.25 * qml.Z(1)
        np.testing.assert_allclose(
            float(device.exact_expval(observable)),
            float(device.exact_expval(qml.Hamiltonian([0.5, 0.25], [qml.Z(0), qml.Z(1)]))),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_trainable_scale_over_a_numpy_coefficient_is_evaluated_and_differentiated(self) -> None:
        results = []
        for device in (NIFDevice(wires=N_QUBITS), qml.device("default.qubit", wires=N_QUBITS)):
            weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
            value = self.expectation(device, qml.s_prod(weight, qml.s_prod(np.array(0.5), WORDS[0]) + WORDS[1]))
            value.backward()
            results.append((float(value), float(weight.grad)))
        np.testing.assert_allclose(results[0], results[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_an_identity_term_with_a_pennylane_numpy_weight_keeps_a_torch_result(self) -> None:
        """A pennylane.numpy offset added to a trainable torch result would fail."""
        values = []
        for device in (NIFDevice(wires=N_QUBITS), qml.device("default.qubit", wires=N_QUBITS)):
            angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
            value = self.expectation(device, qml.Identity(0) + pnp.array(0.3) * WORDS[0], angle)
            assert isinstance(value, torch.Tensor)
            value.backward()
            values.append((float(value), float(angle.grad)))
        np.testing.assert_allclose(values[0], values[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_nested_one_element_weight_keeps_circuit_broadcasting(self) -> None:
        """A one-element weight is read as a scalar, so the output follows the circuit's batch."""

        def build():
            return qml.Hamiltonian(
                [torch.tensor(0.3, dtype=torch.float64)],
                [qml.s_prod(torch.tensor([0.5], dtype=torch.float64), qml.prod(qml.Z(0), qml.Z(2)))],
            )

        angles = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64)
        values = [
            self.expectation(device, build(), angles).detach().numpy()
            for device in (NIFDevice(wires=N_QUBITS), qml.device("default.qubit", wires=N_QUBITS))
        ]
        assert values[0].shape == values[1].shape == (3,)
        np.testing.assert_allclose(values[0], values[1], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_grouping_computed_after_a_first_evaluation_keeps_the_cached_structure_valid(self) -> None:
        """The decomposition is cached per observable, so its words must not change with grouping."""
        device = NIFDevice(wires=N_QUBITS)
        self.expectation(device, qml.Z(0))
        observable = qml.Hamiltonian([0.5, 0.3, 0.2], [qml.Z(0), qml.Z(1) @ qml.Z(2), qml.Z(0)])
        expected = float(device.exact_expval(observable))
        observable.compute_grouping()
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_grouped_hamiltonian_with_a_projector_term_is_evaluated(self) -> None:
        """A grouping cannot be recomputed for a sum with a non-Pauli term, so it is carried over."""

        def build():
            observable = qml.Hamiltonian([0.5, 0.2], [2 * qml.Z(2), qml.Projector([0, 1], wires=[0, 1])])
            observable.grouping_indices = [[0], [1]]
            return observable

        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), build())),
            self.reference(qml.Hamiltonian([0.5, 0.2], [2 * qml.Z(2), qml.Projector([0, 1], wires=[0, 1])])),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_batch_hamiltonian_next_to_another_measurement_is_evaluated(self) -> None:
        """Its nested wires cannot be hashed, so nothing may ask the tape for its wires."""

        @qml.qnode(NIFDevice(wires=N_QUBITS), interface="torch")
        def run():
            self.circuit()
            return qml.expval(BatchHamiltonian([0.5, 0.3], WORDS)), qml.expval(qml.Z(0))

        batch_value, z_value = run()
        np.testing.assert_allclose(
            float(z_value), self.reference(qml.Z(0)), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            float(qml.math.sum(batch_value)),
            self.reference(qml.Hamiltonian([0.5, 0.3], WORDS)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_an_empty_hamiltonian_evaluates_to_zero(self) -> None:
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), qml.Hamiltonian([], []))),
            0.0,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_multiple_of_the_identity_needs_no_device_wire(self) -> None:
        @qml.qnode(NIFDevice(wires=[5, 6, 7]), interface="torch")
        def run():
            return qml.expval(qml.s_prod(0.5, qml.Identity()))

        np.testing.assert_allclose(float(run()), 0.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: qml.sum(qml.Z(0), qml.Z(1), qml.Z(0), grouping_type="qwc"),
            lambda: qml.sum(2 * qml.Z(0), 3 * qml.Z(0), qml.Z(1), grouping_type="qwc"),
            lambda: qml.sum(0.5 * qml.Identity(0), -0.3 * (qml.X(0) @ qml.X(0)), 0.7 * qml.Z(1), grouping_type="qwc"),
        ],
    )
    def test_a_grouped_sum_with_repeated_words_measures_every_term(self, build: Callable) -> None:
        """A sum's grouping indices refer to its merged terms, not its operands, so it keeps its path."""
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), build())),
            self.reference(build()),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "operators",
        [
            [qml.prod(qml.X(0), qml.X(0)), qml.Z(1), qml.X(1)],
            [2.0 * qml.Identity(0), qml.Z(1), qml.X(1)],
            [2.0 * qml.Z(1), qml.Z(1), qml.X(1)],
        ],
    )
    def test_a_grouped_hamiltonian_with_a_special_leading_term_is_evaluated(self, operators: list) -> None:
        """A term equal to the identity or to another term's word, alone in a group, must not leave an
        empty group behind, so the terms are kept exactly as written."""

        def build():
            observable = qml.Hamiltonian([0.5, 0.3, 0.2], operators)
            observable.grouping_indices = [[0], [1, 2]]
            return observable

        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), build())),
            self.reference(qml.Hamiltonian([0.5, 0.3, 0.2], operators)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_term_of_a_hamiltonian_is_read_live_by_the_device(self) -> None:
        """The Hamiltonian keeps its terms as written; a composite term is read live when measured."""
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        observable = qml.Hamiltonian([1.0], [qml.dot(weights, WORDS)])
        device = NIFDevice(wires=N_QUBITS)
        self.expectation(device, observable)
        with torch.no_grad():
            weights[0] = 5.0
        np.testing.assert_allclose(
            float(self.expectation(device, observable)),
            self.reference(qml.dot([5.0, 0.3], WORDS)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "build",
        [
            lambda: qml.Z(0) + qml.Projector([0], wires=[1]) @ qml.Identity(2),
            lambda: 2.0 * (qml.Projector([0], wires=[1]) @ qml.Projector([0], wires=[1])),
            lambda: qml.Z(0) + qml.pow(qml.Projector([0], wires=[1]), 3),
            lambda: qml.Z(0) + (qml.Projector([0], wires=[1]) + qml.Projector([0], wires=[2])) @ qml.Identity(0),
            lambda: qml.Z(0) + qml.Hadamard(1) @ qml.Hadamard(1),
            lambda: (lambda h: qml.Z(0) @ qml.Z(1) + (h - h))(qml.Hermitian(np.diag([1.0, -1.0]), wires=2)),
            lambda: qml.sum(
                qml.Z(0) @ qml.Z(1),
                qml.Hermitian(np.diag([1.0, -1.0]), wires=2),
                -1.0 * qml.Hermitian(np.diag([1.0, -1.0]), wires=2),
            ),
        ],
    )
    def test_a_non_pauli_part_is_evaluated_once_pennylane_has_simplified_it(self, build: Callable) -> None:
        """The device can only measure what PennyLane makes of such a part: ``P @ I`` is ``P``,
        ``H @ H`` is the identity, and a Hermitian that cancels is not measured at all."""
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), build())),
            self.reference(build()),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "build, n_tapes",
        [
            # The projector and Z_0 Z_1 share wire 0; the cancelled X_0 X_1 and Y_1 Y_2 are not measured.
            (
                lambda h: qml.Projector([0], wires=[0]) + ((h + qml.Hamiltonian([0.2], [qml.Z(0) @ qml.Z(1)])) - h),
                2,
            ),
            (lambda h: qml.sum(qml.Projector([0], wires=[0]), 0.5 * qml.Z(1), h - h), 1),
        ],
    )
    def test_a_cancelled_term_next_to_a_non_pauli_part_costs_no_tape(self, build: Callable, n_tapes: int) -> None:
        """``Sum.terms()`` drops a cancelled term once a part with no Pauli representation makes it fall
        back to simplification, so these are the tape counts measured without the rebuild."""
        observable = build(qml.Hamiltonian([0.5, 0.3], [qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Y(2)]))

        @qml.qnode(NIFDevice(wires=N_QUBITS), interface="torch")
        def run():
            self.circuit()
            return qml.expval(observable)

        tapes, _ = qml.workflow.construct_batch(run, level="device")()
        assert len(tapes) == n_tapes
        np.testing.assert_allclose(
            float(run()),
            self.reference(build(qml.Hamiltonian([0.5, 0.3], [qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Y(2)]))),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "build, n_tapes",
        [
            (lambda weight: qml.Z(1) + weight * qml.Hadamard(2), 1),
            (lambda weight: qml.Z(1) + weight * qml.Hermitian(np.diag([1.0, -1.0]), wires=2), 1),
            (
                lambda weight: (
                    qml.Z(1) + qml.Projector([0], wires=[1]) - (weight + 1.0) * qml.Projector([0], wires=[1])
                ),
                1,
            ),
        ],
    )
    def test_a_trainable_term_that_vanishes_is_dropped_as_pennylane_drops_it(
        self, build: Callable, n_tapes: int
    ) -> None:
        """The device cannot measure a Hadamard or a Hermitian at all, and the cancelled projector would
        cost a tape: ``Sum.terms()`` drops all three, so the value is ``<Z_1>`` from one tape."""
        weight = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        observable = build(weight)

        @qml.qnode(NIFDevice(wires=N_QUBITS), interface="torch")
        def run():
            self.circuit()
            return qml.expval(observable)

        tapes, _ = qml.workflow.construct_batch(run, level="device")()
        assert len(tapes) == n_tapes
        np.testing.assert_allclose(
            float(run()), self.reference(qml.Z(1)), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize(
        "operand, reference",
        [
            (lambda: qml.sum(qml.s_prod(0.0, qml.Projector([0], wires=[2])), qml.Z(0)), lambda: qml.Z(0)),
            (lambda: qml.Projector([0], wires=[2]) - qml.Projector([0], wires=[2]), lambda: 0.0 * qml.Z(0)),
            (lambda: qml.Hadamard(0) - qml.Hadamard(0), lambda: 0.0 * qml.Z(0)),
            (
                lambda: 0.5 * qml.Hadamard(0) - 0.5 * qml.Hadamard(0) + qml.Z(0),
                lambda: qml.Z(0),
            ),
        ],
    )
    def test_a_hamiltonian_term_whose_non_pauli_part_vanishes_is_evaluated(
        self, operand: Callable, reference: Callable
    ) -> None:
        """The term reaches the device as written, and the strategies reduce it as ``terms()`` would."""
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), qml.Hamiltonian([0.5, 1.0], [operand(), qml.Z(1)]))),
            self.reference(qml.Hamiltonian([0.5, 1.0], [reference(), qml.Z(1)])),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "square, reference",
        [
            (lambda h: qml.prod(h, h), lambda: 4.0 * qml.Identity(0)),
            (lambda h: qml.pow(h, 2), lambda: 4.0 * qml.Identity(0)),
            (lambda h: 0.5 * qml.prod(h, h), lambda: 2.0 * qml.Identity(0)),
            (
                lambda h: qml.Hamiltonian([1.0, 0.5], [qml.Z(0), qml.prod(h, h)]),
                lambda: qml.Z(0) + 2.0 * qml.Identity(0),
            ),
        ],
    )
    def test_the_square_of_a_hamiltonian_with_a_vanishing_projector_is_evaluated(
        self, square: Callable, reference: Callable
    ) -> None:
        """``(2 Z_1 + 0 P_2)^2 = 4``, but one pass of ``simplify`` leaves ``(2 Z_1) @ (2 Z_1)``, which the
        strategies would read as the identity word with coefficient one."""
        device = NIFDevice(wires=N_QUBITS)
        hamiltonian = qml.Hamiltonian([2.0, 0.0], [qml.Z(1), qml.Projector([0], wires=[2])])
        observable = square(hamiltonian)
        expected = self.reference(reference())
        np.testing.assert_allclose(
            float(self.expectation(device, observable)),
            expected,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_product_with_a_hidden_phase_is_evaluated_through_both_paths(self) -> None:
        """``(2 X_1 X_2) Y_1 Y_2 = -2 Z_1 Z_2``, but ``simplify`` keeps it as a product whose phase ``-1``
        the strategies would drop."""
        hamiltonian = qml.Hamiltonian([2.0, 0.0], [qml.X(1) @ qml.X(2), qml.Projector([0], wires=[0])])
        observable = qml.prod(qml.prod(hamiltonian, qml.Y(1)), qml.Y(2))
        device = NIFDevice(wires=N_QUBITS)
        expected = self.reference(-2.0 * (qml.Z(1) @ qml.Z(2)))
        np.testing.assert_allclose(
            float(self.expectation(device, observable)),
            expected,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_the_square_of_a_scaled_projector_is_evaluated(self) -> None:
        """``(2 P)^2 = 4 P``, which the device measures once ``simplify`` has merged the two scalars."""
        hamiltonian = qml.Hamiltonian([2.0], [qml.Projector([0], wires=[1])])
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), qml.prod(hamiltonian, hamiltonian))),
            self.reference(4.0 * qml.Projector([0], wires=[1])),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("gate", [qml.SX, qml.S, qml.T])
    def test_a_gate_in_a_part_kept_whole_is_measured_through_its_words(self, gate: type) -> None:
        """``Z_0 + i (u + u^dagger) + h.c.`` with ``u = G + 0 P`` is ``Z_0``. Each ``u^dagger`` simplifies to
        ``G^dagger``, which measured as a term of its own would contribute the real part of a
        non-Hermitian expectation instead of cancelling against the words of ``G``."""
        part = qml.sum(gate(1), qml.Hamiltonian([0.0, 0.0], [qml.Z(2), qml.Projector([1], wires=[1])]))
        inner = 1j * (part + qml.adjoint(part))
        observable = qml.Z(0) + inner + qml.adjoint(inner)
        device = NIFDevice(wires=N_QUBITS)
        expected = self.reference(qml.Z(0))
        np.testing.assert_allclose(
            float(self.expectation(device, observable)),
            expected,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.timeout(60)
    def test_a_product_that_simplify_returns_unchanged_is_evaluated(self) -> None:
        """``simplify`` returns ``A @ A @ X_1`` as it is for a Hermitian ``A``, which must end its expansion
        rather than start it again."""
        matrix = np.array([[1.0, 0.5 + 0.5j], [0.5 - 0.5j, -1.0]])
        product = qml.prod(qml.Hermitian(matrix, wires=2), qml.Hermitian(matrix, wires=2), qml.X(1))
        observable = qml.Z(0) + 0.0 * product
        device = NIFDevice(wires=N_QUBITS)
        expected = self.reference(qml.Z(0))
        np.testing.assert_allclose(
            float(self.expectation(device, observable)),
            expected,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_product_that_torch_cannot_simplify_is_evaluated_and_differentiated(self) -> None:
        """``t t^dagger`` with ``t = s A^dagger (I + X_3) B``, a one-element scale ``s = -0.5``, a trainable
        ``A = w_0 Z_0 X_2 Z_3 + 0 H_1`` with ``w_0 = -1.5`` and ``B = 0.75 X_1 Y_3 + 0 P_1``, is
        ``0.6328125 (I - X_3)``, quadratic in ``w_0``. ``simplify`` adds into a copy of a trainable
        weight in place, which torch refuses, so the product is multiplied out. ``BasisState`` on part
        of the wires fails here at HEAD as well, so the state is a product state with
        ``<X_3> = sin(0.9)``."""
        scale = torch.tensor([-0.5], dtype=torch.float32)
        weights = torch.tensor([-1.5, 0.0], dtype=torch.float64, requires_grad=True)
        a = qml.Hamiltonian(weights, [qml.prod(qml.Z(0), qml.X(2), qml.Z(3)), qml.Hadamard(1)])
        b = qml.Hamiltonian([0.75, 0.0], [qml.prod(qml.X(1), qml.Y(3)), qml.Projector([0], wires=[1])])
        part = scale * qml.prod(qml.adjoint(a), qml.Identity(3) + qml.X(3), b)
        observable = qml.prod(part, qml.adjoint(part))
        amplitudes = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [np.cos(0.45), np.sin(0.45)]], dtype=complex)
        device = NIFDevice(wires=4)

        @qml.qnode(device, interface="torch")
        def run():
            ProductState(amplitudes, wires=range(4))
            return qml.expval(observable)

        expected = 0.6328125 * (1.0 - np.sin(0.9))
        value = run()
        (gradient,) = torch.autograd.grad(value, weights)
        np.testing.assert_allclose(float(value), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(
            float(gradient[0]), 2.0 * expected / -1.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_a_product_that_autograd_cannot_simplify_is_evaluated(self) -> None:
        """``u u^dagger = 6.25 I`` with ``u = SWAP (-2 Y_1 Z_2 + 1.5 Y_0 X_1 + 0 P)`` in pennylane.numpy
        weights, which numpy refuses to simplify with a ``TypeError``. HEAD evaluates it in a QNode. Built
        with ``qml.prod`` instead, the product reduces to the identity alone and fails inside PennyLane, at
        HEAD as well."""
        weights = pnp.array([-2.0, 1.5, 0.0])
        hamiltonian = qml.Hamiltonian(
            weights, [qml.Y(1) @ qml.Z(2), qml.Y(0) @ qml.X(1), qml.Projector([1, 1], wires=[1, 2])]
        )
        part = qml.SWAP(wires=[1, 2]) @ hamiltonian
        observable = part @ qml.adjoint(part)
        device = NIFDevice(wires=N_QUBITS)
        np.testing.assert_allclose(
            float(self.expectation(device, observable)), 6.25, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            float(device.exact_expval(observable)), 6.25, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_a_vanishing_weight_in_a_projector_product_is_measured_as_pennylane_splits_it(
        self, requires_grad: bool
    ) -> None:
        """Simplifying the caller's own product would make PennyLane split it into a projector product,
        which the device cannot measure, instead of the projector it measures with a zero weight."""
        weight = torch.tensor(0.0, dtype=torch.float64, requires_grad=requires_grad)
        observable = qml.Projector([1], wires=[0]) @ qml.s_prod(weight, qml.X(1) @ qml.X(2))
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), observable)),
            0.0,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_dot_vector_is_not_written_into_when_a_part_is_simplified(self) -> None:
        """The two projector terms merge into one: PennyLane would add them into the caller's vector."""
        coefficients = torch.tensor([-0.75, -0.3], dtype=torch.float64)
        projector = qml.Projector([1], wires=[1])
        observable = qml.prod(qml.dot(coefficients, [qml.Z(0) + projector, projector]), qml.Identity(2))
        value = float(self.expectation(NIFDevice(wires=N_QUBITS), observable))
        np.testing.assert_array_equal(coefficients.numpy(), [-0.75, -0.3])
        np.testing.assert_allclose(
            value,
            self.reference(-0.75 * qml.Z(0) - 1.05 * qml.Projector([1], wires=[1])),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_single_precision_weight_leaves_the_constants_exact_on_a_double_precision_device(self) -> None:
        """PennyLane would multiply each constant in its own precision; one stacked array must stay double."""
        weight = torch.tensor(0.3, dtype=torch.float32)
        constant = 1.2345678901234
        np.testing.assert_allclose(
            float(self.expectation(NIFDevice(wires=N_QUBITS), weight * qml.Z(0) + constant * qml.Z(1))),
            self.reference(qml.dot([float(weight), constant], [qml.Z(0), qml.Z(1)])),
            atol=DOUBLE_PRECISION_ATOL,
            rtol=0.0,
        )

    @pytest.mark.parametrize("scale", [lambda h: -h, lambda h: qml.s_prod(2.0, h)])
    def test_a_scaled_hamiltonian_with_a_nested_hamiltonian_keeps_one_tape(self, scale: Callable) -> None:
        """``SProd.terms()`` reads the Hamiltonian's own terms, measuring the nested one whole; its words
        overlap on wire 1, so flattening them would take a second tape."""

        def build() -> qml.operation.Operator:
            nested = qml.Hamiltonian([0.25, -0.8], [qml.Y(1) @ qml.X(2), qml.X(1) @ qml.Y(2)])
            return scale(qml.Hamiltonian([0.11, 1.0], [qml.X(0), nested]))

        observable = build()

        @qml.qnode(NIFDevice(wires=N_QUBITS), interface="torch")
        def run():
            self.circuit()
            return qml.expval(observable)

        tapes, _ = qml.workflow.construct_batch(run, level="device")()
        assert len(tapes) == 1
        np.testing.assert_allclose(
            float(run()), self.reference(build()), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize(
        "build",
        [
            lambda: WORDS[0] + qml.Z(2),
            lambda: 0.5 * (WORDS[0] + qml.Z(2)),
            lambda: 1.234 * qml.Z(0) + 0.777 * qml.Z(1),
            lambda: qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS),
            lambda: qml.sum(qml.Hamiltonian(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS), qml.Z(2)),
            lambda: 0.5 * qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS),
            lambda: qml.adjoint(qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS)),
            lambda: qml.Projector([0], wires=[0]) + 0.5 * qml.Z(1),
            lambda: qml.Hamiltonian([0.5, 0.3], WORDS, grouping_type="qwc"),
            lambda: qml.Hamiltonian(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS, grouping_type="qwc"),
        ],
    )
    def test_a_single_precision_device_returns_single_precision(self, build: Callable) -> None:
        """A rebuilt observable keeps the result in the device's precision, so it can feed a
        single-precision model."""
        value = self.expectation(NIFDevice(wires=N_QUBITS, r_dtype=torch.float32, c_dtype=torch.complex64), build())
        assert value.dtype == torch.float32
        np.testing.assert_allclose(
            float(value), self.reference(build()), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_double_precision_weights_keep_double_precision_on_a_single_precision_device(self) -> None:
        """As PennyLane multiplies them in: the weights' precision wins over the device's."""
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        device = NIFDevice(wires=N_QUBITS, r_dtype=torch.float32, c_dtype=torch.complex64)
        assert self.expectation(device, qml.dot(weights, WORDS)).dtype == torch.float64

    def test_the_adjoint_of_a_batched_batch_hamiltonian_is_evaluated_for_every_batch_entry(self) -> None:
        """Row ``i`` of the coefficients holds the weights of term ``i`` across the batch."""
        weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        value = self.expectation(NIFDevice(wires=N_QUBITS), qml.adjoint(BatchHamiltonian(weights, WORDS)))
        np.testing.assert_allclose(
            np.ravel(qml.math.unwrap(value)),
            [self.reference(qml.Hamiltonian(column, WORDS)) for column in weights.T],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
