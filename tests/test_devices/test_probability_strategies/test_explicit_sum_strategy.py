import numpy as np
import pennylane as qml
import pytest
from pennylane.wires import Wires

from matchcake import utils
from matchcake.devices.probability_strategies import ExplicitSumStrategy
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.operations.state_preparation import (
    MinusState,
    PlusState,
    ProductState,
    StatePrepFromGates,
)


class TestExplicitSumStrategy:
    @pytest.fixture
    def strategy(self):
        return ExplicitSumStrategy()

    @pytest.fixture
    def two_qubit_setup(self):
        num_wires = 2
        wires = np.arange(num_wires)
        global_sptm = SingleParticleTransitionMatrixOperation.random(wires=wires, seed=42)
        transition_matrix = utils.make_transition_matrix_from_action_matrix(global_sptm.matrix())
        state_prep_op = qml.BasisState(np.zeros(num_wires, dtype=int), wires)
        return wires, transition_matrix, state_prep_op

    def test_wires_as_int(self, strategy, two_qubit_setup):
        wires, transition_matrix, state_prep_op = two_qubit_setup
        result = strategy(
            state_prep_op=state_prep_op,
            target_binary_states=np.array([0]),
            wires=0,
            all_wires=Wires(wires),
            transition_matrix=transition_matrix,
        )
        assert result is not None

    def test_compute_single_wires_as_int(self, strategy, two_qubit_setup):
        wires, transition_matrix, state_prep_op = two_qubit_setup
        result = strategy._compute_single(
            state_prep_op=state_prep_op,
            target_binary_state=np.array([0]),
            wires=0,
            all_wires=Wires(wires),
            transition_matrix=transition_matrix,
        )
        assert result is not None

    def test_can_execute_basis_state_true(self, strategy):
        assert strategy.can_execute(qml.BasisState(np.zeros(2, dtype=int), wires=[0, 1])) is True

    def test_can_execute_non_state_false(self, strategy):
        assert strategy.can_execute(qml.PauliX(0)) is False

    def test_can_execute_unbatched_basis_product_state_true(self, strategy):
        prep = ProductState.from_basis_state(np.array([1, 0]), wires=[0, 1])
        assert strategy.can_execute(prep) is True

    def test_can_execute_batched_basis_product_state_false(self, strategy):
        # The explicit sum contracts the Majorana decomposition of a single system state,
        # so a batched preparation must fall through rather than raise downstream.
        prep = ProductState.from_basis_state(np.array([[0, 0], [1, 1]]), wires=[0, 1])
        assert prep.batch_size == 2
        assert strategy.can_execute(prep) is False

    def test_can_execute_non_basis_product_state_false(self, strategy):
        amplitudes = np.full((2, 2), 1.0 / np.sqrt(2), dtype=complex)
        prep = ProductState(amplitudes, wires=[0, 1])
        assert strategy.can_execute(prep) is False

    def test_can_execute_basis_state_prep_from_gates_true(self, strategy):
        prep = StatePrepFromGates(lambda wires: (qml.X(w) for w in wires), wires=[0, 1])
        assert strategy.can_execute(prep) is True

    @pytest.mark.parametrize("prep_class", [PlusState, MinusState])
    def test_can_execute_non_basis_state_prep_from_gates_false(self, strategy, prep_class):
        # ``StatePrepFromGates`` is a ``ProductState`` subclass, so accepting it
        # unconditionally would send a non-basis preparation into a path that cannot
        # represent it. It must fall through to ProductStateProbabilityStrategy instead.
        assert strategy.can_execute(prep_class(wires=[0, 1])) is False

    @pytest.mark.parametrize("prep_class", [PlusState, MinusState])
    def test_can_execute_false_implies_no_system_basis_state(self, strategy, prep_class):
        # The load-bearing invariant: ``can_execute`` and the basis-state extraction must
        # agree, so that a preparation accepted by the former is always convertible.
        prep = prep_class(wires=[0, 1])
        assert strategy.can_execute(prep) is False
        with pytest.raises(ValueError):
            strategy.system_basis_state_from_state_prep_op(prep)
