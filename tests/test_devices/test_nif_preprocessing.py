from typing import Callable

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane import numpy as pnp

from matchcake import BatchHamiltonian
from matchcake.devices.nif_preprocessing import (
    _in_working_precision,
    _restored,
    live_observable,
    nif_split_non_commuting,
)

from ..configs import ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON

N_QUBITS = 3

# Words with distinct supports, used to build the observables below.
WORDS = [qml.Z(0) @ qml.Z(1), qml.X(1) @ qml.X(2)]


class TestNifPreprocessing:
    @pytest.mark.parametrize(
        "observable",
        [
            qml.Z(0),
            qml.prod(qml.Z(0), qml.X(1)),
            qml.Hamiltonian(np.array([0.5, 0.3]), WORDS),
            qml.Hamiltonian(torch.tensor([0.5, 0.3], dtype=torch.float64), WORDS),
            BatchHamiltonian([0.5, 0.3], WORDS),
            qml.Hamiltonian(np.array([2.0]), [BatchHamiltonian([0.5, 0.3], WORDS)]),
            qml.Hermitian(np.diag([1.0, -1.0]), wires=0),
            0.7 * qml.Identity(0),
            qml.Hamiltonian(torch.tensor([0.7], dtype=torch.float64), [0.5 * qml.Identity(0)]),
            qml.sum(qml.Hamiltonian([], []), qml.Hamiltonian([], [])),
            qml.Hamiltonian(np.array([0.5]), [-qml.prod(qml.Z(0), qml.Z(1))]),
            qml.Hamiltonian(np.array([1.0]), [qml.Z(0) + qml.X(0)]),
            (lambda projector: projector - projector)(qml.Projector([0], wires=[0])),
            qml.s_prod(np.float64(2.0), qml.Hamiltonian(np.array([0.5, 0.3]), WORDS)),
            qml.s_prod(torch.tensor(0.5, dtype=torch.float64), qml.Z(0) @ qml.Z(1)),
            (lambda h: qml.sum(qml.pow(qml.Y(2), 0), h - h))(qml.Hamiltonian([0.5], [qml.X(1)])),
        ],
    )
    def test_an_observable_that_needs_no_rebuild_is_measured_as_it_is(self, observable: qml.operation.Operator) -> None:
        """Anything already read live, or that PennyLane handles through another path, keeps the path it
        took before: plain words; ungrouped Hamiltonians, a ``BatchHamiltonian`` among them, whose terms
        are measured as written and read live by the device; a scalar multiple of a Hamiltonian or of a
        word, read through the base's own ``terms()``; an observable with no Pauli part; and one that
        reduces to a multiple of the identity or to nothing once the terms PennyLane drops are dropped
        (an offset to the splitter)."""
        assert live_observable(observable) is observable

    @pytest.mark.parametrize(
        "observable",
        [
            0.5 * qml.Z(0) + qml.X(1),
            qml.adjoint(qml.Hamiltonian([0.5, 0.3], WORDS)),
            qml.Hermitian(np.diag([1.0, -1.0]), wires=0) + qml.Z(1),
            qml.Z(0) + qml.Projector([0], wires=[1]) @ qml.Identity(2),
            qml.Z(0) + qml.Hadamard(1) @ qml.Hadamard(1),
            qml.Projector([0], wires=[1]) @ qml.Identity(2),
            qml.Hamiltonian([0.5, 0.3], WORDS, grouping_type="qwc"),
            qml.Hamiltonian([0.5, 0.3], [2.0 * WORDS[0], WORDS[1]], grouping_type="commuting"),
            (lambda shared: qml.sum(shared, 0.5 * shared))(qml.Z(0) @ qml.Z(1)),
            qml.sum(qml.Z(0), qml.Z(1), qml.Z(0), grouping_type="qwc"),
            qml.Hamiltonian([0.5, 0.3], WORDS),
            qml.Hamiltonian([0.5], [-qml.prod(qml.Z(0), qml.Z(1))]),
            qml.Hamiltonian([1.0], [qml.Z(0) + qml.X(0)]),
            qml.s_prod(2.0, qml.Hamiltonian([0.5, 0.3], WORDS)),
            -qml.Hamiltonian([0.5, 0.3], WORDS),
            0.5 * (qml.Z(0) @ qml.Z(1)),
            qml.sum(0.5 * qml.Identity(0), -0.3 * (qml.X(0) @ qml.X(0)), 0.7 * qml.Z(1), grouping_type="qwc"),
        ],
    )
    def test_a_composite_or_grouped_observable_is_rebuilt_as_a_hamiltonian(
        self, observable: qml.operation.Operator
    ) -> None:
        rebuilt = live_observable(observable)
        assert isinstance(rebuilt, qml.ops.LinearCombination)
        np.testing.assert_allclose(
            qml.matrix(rebuilt, wire_order=range(N_QUBITS)),
            qml.matrix(observable, wire_order=range(N_QUBITS)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_precomputed_grouping_is_kept(self) -> None:
        """All the ``Z Z`` words commute qubit-wise, so the user's grouping measures them in one tape."""
        words = [qml.Z(first) @ qml.Z(second) for first in range(N_QUBITS) for second in range(first + 1, N_QUBITS)]
        observable = qml.Hamiltonian([0.1] * len(words), words, grouping_type="qwc")
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))], [qml.expval(observable)]
        )
        tapes, _ = nif_split_non_commuting(tape)
        assert len(tapes) == 1

    def test_a_grouping_is_kept_through_a_rebuild(self) -> None:
        """The splitter measures the user's groups: rebuilding term by term keeps their indices valid."""
        observable = qml.Hamiltonian(
            [0.5, 0.2, 0.1],
            [2.0 * (qml.X(0) @ qml.X(1)), qml.Y(0) @ qml.Y(1), qml.Z(0) @ qml.Z(1)],
            grouping_type="commuting",
        )
        rebuilt = live_observable(observable)
        assert rebuilt is not observable
        assert rebuilt.grouping_indices == observable.grouping_indices
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))], [qml.expval(observable)]
        )
        tapes, _ = nif_split_non_commuting(tape)
        assert len(tapes) == len(observable.grouping_indices)

    def test_a_grouped_sum_keeps_its_grouping(self) -> None:
        observable = qml.sum(qml.Z(0), qml.Z(0) @ qml.Z(1), qml.Z(1) @ qml.Z(2), grouping_type="qwc")
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))], [qml.expval(observable)]
        )
        tapes, _ = nif_split_non_commuting(tape)
        assert len(tapes) == 1
        assert live_observable(observable).grouping_indices == observable.grouping_indices

    def test_a_grouped_sum_is_rebuilt_from_its_current_coefficients(self) -> None:
        weights = torch.tensor([0.5, 0.3], dtype=torch.float64)
        observable = qml.sum(weights[0] * WORDS[0], weights[1] * WORDS[1], grouping_type="qwc")
        with torch.no_grad():
            weights.mul_(-2.0)
        rebuilt = live_observable(observable)
        np.testing.assert_allclose(
            qml.matrix(rebuilt, wire_order=range(N_QUBITS)),
            qml.matrix(qml.Hamiltonian([-1.0, -0.6], WORDS), wire_order=range(N_QUBITS)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_word_cancelled_when_a_grouped_sum_was_built_gets_a_group_of_its_own(self) -> None:
        """The cache drops ``Z_0``, whose weights cancel at build time, so no group holds it."""
        weights = torch.tensor([0.5, -0.5, 0.3], dtype=torch.float64)
        observable = qml.sum(weights[0] * qml.Z(0), weights[1] * qml.Z(0), weights[2] * qml.X(1))
        observable.compute_grouping()
        with torch.no_grad():
            weights[1] = 0.2
        rebuilt = live_observable(observable)
        assert sorted(index for group in rebuilt.grouping_indices for index in group) == list(range(len(rebuilt.ops)))
        np.testing.assert_allclose(
            qml.matrix(rebuilt, wire_order=range(N_QUBITS)),
            qml.matrix(qml.Hamiltonian([0.7, 0.3], [qml.Z(0), qml.X(1)]), wire_order=range(N_QUBITS)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_trainable_scalar_of_a_grouped_hamiltonian_term_is_folded_into_its_coefficient(self) -> None:
        weight = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        observable = qml.Hamiltonian([0.5, 0.2], [qml.s_prod(weight, qml.Z(0)), qml.X(1)], grouping_type="qwc")
        rebuilt = live_observable(observable)
        assert rebuilt.grouping_indices == observable.grouping_indices
        assert [type(operator) for operator in rebuilt.ops] == [qml.Z, qml.X]
        (gradient,) = torch.autograd.grad(rebuilt.coeffs[0], weight)
        np.testing.assert_allclose(
            (float(rebuilt.coeffs[0].detach()), float(rebuilt.coeffs[1].detach()), float(gradient)),
            (-0.15, 0.2, 0.5),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_an_ungrouped_hamiltonian_with_a_trainable_term_scalar_is_rebuilt(self) -> None:
        weight = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        rebuilt = live_observable(qml.Hamiltonian([0.5], [qml.s_prod(weight, qml.Z(0))]))
        assert [type(operator) for operator in rebuilt.ops] == [qml.Z]
        (gradient,) = torch.autograd.grad(rebuilt.coeffs[0], weight)
        np.testing.assert_allclose(float(gradient), 0.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_trainable_coefficient_nested_in_a_hamiltonian_term_is_found(self) -> None:
        weight = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        nested = qml.adjoint(qml.Hamiltonian(torch.stack([weight]), [qml.Z(0)]))
        rebuilt = live_observable(qml.Hamiltonian([0.5], [nested]))
        assert [type(operator) for operator in rebuilt.ops] == [qml.Z]
        (gradient,) = torch.autograd.grad(rebuilt.coeffs[0], weight)
        np.testing.assert_allclose(float(gradient), 0.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_grouped_sum_without_pauli_terms_is_measured_as_it_is(self) -> None:
        observable = qml.sum(qml.Projector([0], wires=[0]), qml.Projector([1], wires=[1]))
        observable._grouping_indices = ((0,), (1,))
        assert live_observable(observable) is observable

    @pytest.mark.parametrize(
        "coefficient",
        [
            torch.tensor([1.0], dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, dtype=torch.complex128, requires_grad=True),
        ],
    )
    def test_a_restored_result_keeps_its_value_and_shape(self, coefficient: torch.Tensor) -> None:
        result = torch.tensor(0.25, dtype=torch.float32)
        restored = torch.as_tensor(_restored(result, coefficient, 0))
        assert restored.shape == result.shape and restored.requires_grad
        np.testing.assert_allclose(restored.detach().real.numpy(), 0.25, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize(
        "r_dtype, weight_dtype, dtype",
        [
            (torch.float32, torch.complex64, torch.complex64),
            (torch.float32, torch.complex128, torch.complex64),
            (torch.float64, torch.complex64, torch.complex128),
            (torch.float32, torch.float64, torch.float32),
            (torch.float64, torch.float32, torch.float64),
        ],
    )
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_a_torch_multiple_of_the_identity_is_returned_in_the_precision_of_the_device(
        self, r_dtype: torch.dtype, weight_dtype: torch.dtype, dtype: torch.dtype, requires_grad: bool
    ) -> None:
        value = 0.5 + 0.25j if weight_dtype.is_complex else 0.5
        weight = torch.tensor(value, dtype=weight_dtype, requires_grad=requires_grad)
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))],
            [qml.expval(qml.s_prod(weight, qml.Identity(0)))],
        )
        tapes, postprocessing = nif_split_non_commuting(tape, r_dtype=r_dtype)
        result = postprocessing(())
        assert len(tapes) == 0 and result.dtype == dtype and result.requires_grad is requires_grad
        np.testing.assert_allclose(complex(result.detach()), value, atol=1e-6)

    def test_unit_coefficients_are_left_to_pennylane_with_partitioned_shots(self) -> None:
        weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))],
            [qml.expval(qml.s_prod(weight, qml.Z(0)))],
            shots=(10, 10),
        )
        _, postprocessing = nif_split_non_commuting(tape)
        results = ((torch.tensor(1.0), torch.tensor(-1.0)),)
        assert not any(value.requires_grad for value in postprocessing(results))

    @pytest.mark.parametrize(
        "build",
        [
            lambda coefficients: qml.Hamiltonian(coefficients, WORDS),
            lambda coefficients: qml.Hamiltonian(coefficients, WORDS, grouping_type="qwc"),
            lambda coefficients: qml.s_prod(1.0, qml.Hamiltonian(coefficients, WORDS)),
        ],
    )
    @pytest.mark.parametrize("r_dtype, dtype", [(None, np.float64), (torch.float32, np.float32)])
    def test_python_coefficients_are_given_as_an_array_in_the_precision_of_the_device(
        self, build: Callable, r_dtype, dtype
    ) -> None:
        """Torch reads a python float in its default precision when PennyLane multiplies it into a result."""
        rebuilt = live_observable(build([0.1234567891234567, 0.7654321987654321]), r_dtype)
        assert type(rebuilt.coeffs) is np.ndarray and rebuilt.coeffs.dtype == dtype
        np.testing.assert_allclose(rebuilt.coeffs, [0.1234567891234567, 0.7654321987654321], rtol=1e-6)

    def test_a_grouped_hamiltonian_term_that_expands_into_several_words_is_kept_whole(self) -> None:
        weight = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        observable = qml.Hamiltonian([0.5, 0.2], [qml.s_prod(weight, qml.Z(0) + qml.X(0)), qml.Z(1)])
        observable.grouping_indices = [[0], [1]]
        rebuilt = live_observable(observable)
        assert rebuilt.grouping_indices == observable.grouping_indices
        assert isinstance(rebuilt.ops[0], qml.ops.op_math.Sum)
        (gradient,) = torch.autograd.grad(rebuilt.coeffs[0], weight)
        np.testing.assert_allclose(float(gradient), 0.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_a_grouped_hamiltonian_term_without_pauli_words_keeps_its_operator(self) -> None:
        weight = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        observable = qml.Hamiltonian([0.5, 0.2], [qml.s_prod(weight, qml.Projector([0], wires=[0])), qml.Z(1)])
        observable.grouping_indices = [[0], [1]]
        rebuilt = live_observable(observable)
        assert isinstance(rebuilt.ops[0], qml.Projector)
        (gradient,) = torch.autograd.grad(rebuilt.coeffs[0], weight)
        np.testing.assert_allclose(float(gradient), 0.5, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_an_observable_with_no_terms_is_measured_as_it_is(self) -> None:
        observable = qml.sum(qml.Hamiltonian([], []), qml.Hamiltonian([], []))
        assert live_observable(observable) is observable

    @pytest.mark.parametrize(
        "operators, grouping_indices",
        [
            ([qml.Z(0) + qml.X(1), qml.Z(2)], [[0], [1]]),
            ([qml.prod(qml.X(0), qml.X(0)), qml.Z(1), qml.X(1)], [[0], [1, 2]]),
            ([2.0 * qml.Z(0), qml.Z(0), qml.Z(1)], [[0], [1, 2]]),
        ],
    )
    def test_a_grouped_hamiltonian_gets_fresh_coefficients_and_keeps_its_terms(
        self, operators: list, grouping_indices: list
    ) -> None:
        """Its grouping indices refer to its terms as written, so the terms are kept exactly: one that
        expands to several words, or reduces to the identity or to another term's word, would
        otherwise empty a group or shift the indices."""
        weights = torch.tensor([0.5, 0.3, 0.2][: len(operators)], dtype=torch.float64)
        observable = qml.Hamiltonian(weights, operators)
        observable.grouping_indices = grouping_indices
        rebuilt = live_observable(observable)
        assert rebuilt is not observable
        assert all(new is old for new, old in zip(rebuilt.ops, observable.ops))
        assert rebuilt.grouping_indices == observable.grouping_indices
        assert rebuilt.coeffs is not weights

    def test_an_empty_grouped_hamiltonian_is_measured_as_it_is(self) -> None:
        """PennyLane stores an empty grouping as ``()``, which is a grouping all the same."""
        observable = qml.Hamiltonian([], [])
        observable.grouping_indices = []
        assert live_observable(observable) is observable

    def test_a_lone_batch_hamiltonian_tape_is_passed_through(self) -> None:
        """A ``BatchHamiltonian`` is evaluated term by term by the device, so its tape is not split."""
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))],
            [qml.expval(BatchHamiltonian([0.5, 0.3], WORDS))],
        )
        tapes, postprocessing = nif_split_non_commuting(tape)
        assert tapes == [tape]
        assert postprocessing(["result"]) == "result"

    def test_a_measurement_other_than_an_expectation_is_left_as_it_is(self) -> None:
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))],
            [qml.probs(wires=[0, 1]), qml.expval(0.5 * qml.Z(0) + qml.X(1))],
        )
        tapes, _ = nif_split_non_commuting(tape)
        assert any(
            isinstance(measurement, qml.measurements.ProbabilityMP)
            for split in tapes
            for measurement in split.measurements
        )

    def test_a_tape_with_nothing_to_rebuild_measures_the_original_observables(self) -> None:
        observable = qml.Hamiltonian(np.array([0.5, 0.3]), WORDS)
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))], [qml.expval(observable)]
        )
        tapes, _ = nif_split_non_commuting(tape)
        measured = [measurement.obs for split in tapes for measurement in split.measurements]
        assert all(any(word is operator for operator in observable.ops) for word in measured)

    def test_a_hamiltonian_of_a_nested_sum_keeps_one_tape(self) -> None:
        """Flattening it into words would put overlapping words on separate tapes."""
        observable = qml.Hamiltonian(
            [1.0], [qml.dot([0.5, 0.3, 0.2], [qml.Z(1), qml.Y(1) @ qml.Y(2), qml.X(1) @ qml.X(2)])]
        )
        tape = qml.tape.QuantumScript(
            [qml.BasisState(np.array([0, 1, 0]), wires=range(N_QUBITS))], [qml.expval(observable)]
        )
        tapes, _ = nif_split_non_commuting(tape)
        assert len(tapes) == 1

    @pytest.mark.parametrize(
        "observable, kind",
        [
            (qml.Z(0) + qml.Projector([0], wires=[1]) @ qml.Identity(2), qml.Projector),
            (qml.Z(0) + qml.pow(qml.Projector([0], wires=[1]), 3), qml.Projector),
            (qml.Z(0) + (2.0 * qml.Projector([0], wires=[1])) @ qml.Identity(2), qml.Projector),
            (
                qml.Z(0) + (qml.Projector([0], wires=[1]) + qml.Projector([0], wires=[2])) @ qml.Identity(0),
                qml.Projector,
            ),
            (qml.Z(0) + qml.Hadamard(1) @ qml.Hadamard(1), qml.Identity),
        ],
    )
    def test_a_non_pauli_part_is_measured_as_pennylane_simplifies_it(
        self, observable: qml.operation.Operator, kind: type
    ) -> None:
        """The device measures a projector, not a product, power, scalar multiple or sum of them."""
        rebuilt = live_observable(observable)
        assert any(isinstance(operator, kind) for operator in rebuilt.ops)
        assert not any(
            isinstance(operator, (qml.ops.Prod, qml.ops.Pow, qml.ops.SProd, qml.ops.Sum)) for operator in rebuilt.ops
        )

    def test_parts_that_simplify_to_the_same_operator_are_merged(self) -> None:
        """``P @ I`` and ``-P`` cancel once simplified, as ``Sum.terms()`` merges them by hash."""
        observable = qml.sum(
            qml.Z(0), qml.Projector([0], wires=[1]) @ qml.Identity(2), -1.0 * qml.Projector([0], wires=[1])
        )
        rebuilt = live_observable(observable)
        assert not any(isinstance(operator, qml.Projector) for operator in rebuilt.ops)

    @pytest.mark.parametrize(
        "observable, dropped",
        [
            (
                (lambda h: qml.sum(qml.Projector([0], wires=[0]), 0.5 * qml.Z(1), h - h))(
                    qml.Hamiltonian([0.5], [qml.X(1)])
                ),
                True,
            ),
            ((lambda h: qml.sum(qml.Z(0), 0.5 * qml.Z(1), h - h))(qml.Hamiltonian([0.5], [qml.X(1)])), False),
            (
                qml.Projector([0], wires=[0]) + torch.tensor(0.0, dtype=torch.float64, requires_grad=True) * qml.X(1),
                False,
            ),
            (
                qml.Projector([0], wires=[0]) + torch.tensor(0.0, dtype=torch.float64) * qml.X(1),
                True,
            ),
        ],
    )
    def test_a_cancelled_term_is_dropped_only_where_pennylane_drops_it(
        self, observable: qml.operation.Operator, dropped: bool
    ) -> None:
        """``Sum.terms()`` drops it when it falls back to simplification, which a part with no Pauli
        representation makes it do. A measurable term of trainable weight is kept for its gradient."""
        rebuilt = live_observable(observable)
        assert any(isinstance(operator, qml.X) for operator in rebuilt.ops) is not dropped

    @pytest.mark.parametrize(
        "observable, r_dtype, dtype",
        [
            (0.5 * qml.Z(0) + qml.X(1), None, "float64"),
            (0.5 * qml.Z(0) + qml.X(1), torch.float32, "float32"),
            (qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS), None, "float64"),
            (qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS), torch.float64, "float64"),
            (qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS), torch.float32, "float32"),
            (
                torch.tensor(0.5, dtype=torch.float32) * qml.Z(0) + torch.tensor(0.3, dtype=torch.float64) * qml.X(1),
                torch.float32,
                "float64",
            ),
            (torch.tensor(0.5, dtype=torch.float32) * (qml.X(0) @ qml.Y(0)), torch.float32, "complex64"),
            (
                (lambda shared: qml.sum(shared, shared))(qml.dot(torch.tensor([0.5, 0.3], dtype=torch.float32), WORDS)),
                torch.float32,
                "float32",
            ),
        ],
    )
    def test_rebuilt_coefficients_take_the_working_precision(
        self, observable: qml.operation.Operator, r_dtype, dtype: str
    ) -> None:
        """Only a single-precision device lowers them: torch ones when all the caller's torch
        coefficients are single precision, numpy ones always. Double precision keeps every constant exact."""
        rebuilt = live_observable(observable, r_dtype)
        assert str(rebuilt.coeffs.dtype).rsplit(".", 1)[-1] == dtype

    @pytest.mark.parametrize(
        "coefficients",
        [
            np.array([0.5, 0.3]),
            pnp.array([0.5, 0.3], requires_grad=True),
            torch.tensor([0.5, 0.3], dtype=torch.float32),
        ],
    )
    def test_a_grouped_hamiltonian_copy_keeps_the_type_and_precision_of_its_coefficients(self, coefficients) -> None:
        observable = qml.Hamiltonian(coefficients, WORDS, grouping_type="qwc")
        rebuilt = live_observable(observable)
        assert type(rebuilt.coeffs) is type(observable.coeffs)
        assert rebuilt.coeffs is not observable.coeffs
        assert getattr(rebuilt.coeffs, "dtype", None) == getattr(observable.coeffs, "dtype", None)
        assert qml.math.requires_grad(rebuilt.coeffs) == qml.math.requires_grad(observable.coeffs)

    def test_a_composite_holding_a_batch_hamiltonian_is_rebuilt_from_its_words(self) -> None:
        rebuilt = live_observable(qml.sum(BatchHamiltonian([0.5, 0.3], WORDS), qml.Z(2)))
        assert not isinstance(rebuilt, BatchHamiltonian)
        assert all(isinstance(operator, (qml.ops.Prod, qml.Z)) for operator in rebuilt.ops)
        np.testing.assert_allclose(
            rebuilt.coeffs, [0.5, 0.3, 1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_an_autograd_array_of_coefficients_is_left_to_pennylane(self) -> None:
        """Casting a pennylane.numpy array on a single-precision device would cut it from autograd."""
        coefficients = pnp.array([0.5, 0.3], requires_grad=True)
        assert _in_working_precision(coefficients, qml.Z(0), torch.float32) is coefficients
