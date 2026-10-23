import numpy as np
import pennylane as qml
import pytest

from matchcake.devices.expval_strategies.expval_from_probabilities import (
    ExpvalFromProbabilitiesStrategy,
)
from matchcake.devices.expval_strategies.terms_splitter import TermsSplitter

from ...configs import ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON


class TestTermsSplitter:
    @pytest.fixture
    def strategy(self):
        return TermsSplitter([ExpvalFromProbabilitiesStrategy()])

    def test_on_non_valid_observable(self, strategy):
        with pytest.raises(ValueError):
            strategy(
                qml.BasisState([0, 0], [0, 1]),
                qml.Z(0) @ qml.X(1),
                prob=np.random.random(4),
            )

    def test_format_observable(self, strategy):
        hamiltonian = strategy._format_observable(qml.X(0))
        assert isinstance(hamiltonian, qml.Hamiltonian)
        np.testing.assert_allclose(hamiltonian.coeffs, [1.0])

    def test_split_returns_per_strategy_lists(self, strategy):
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        observable = qml.Z(0)
        splits = strategy.split(state_prep_op, observable)
        assert len(splits) == 1
        assert len(splits[0]) >= 1

    def test_call_on_valid_observable(self, strategy):
        from matchcake import NIFDevice
        from matchcake.operations import CompZX

        nif_device = NIFDevice(wires=2)
        nif_device.apply_generator([CompZX(wires=[0, 1])])
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        nif_device.apply_state_prep(state_prep_op)
        observable = qml.Z(0) @ qml.I(1)
        result = strategy(state_prep_op, observable, prob_func=nif_device.probability)
        assert np.asarray(result).ndim <= 1

    def test_each_coefficient_is_applied_once(self, strategy):
        """The sub-strategies already read the scalar of a term, so weighting it again squared it."""
        from matchcake import NIFDevice
        from matchcake.operations import CompZX

        nif_device = NIFDevice(wires=2)
        nif_device.apply_generator([CompZX(wires=[0, 1])])
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        nif_device.apply_state_prep(state_prep_op)
        word_values = [
            ExpvalFromProbabilitiesStrategy()(state_prep_op, word, prob_func=nif_device.probability)
            for word in (qml.Z(0), qml.Z(1))
        ]
        result = strategy(state_prep_op, 0.5 * qml.Z(0) + 0.25 * qml.Z(1), prob_func=nif_device.probability)
        np.testing.assert_allclose(
            np.ravel(result),
            np.ravel(0.5 * word_values[0] + 0.25 * word_values[1]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_term_accepted_by_several_strategies_is_counted_once(self):
        """Each term goes to the first strategy that accepts it, not to every one."""
        from matchcake import NIFDevice

        nif_device = NIFDevice(wires=3)

        @qml.qnode(nif_device)
        def circuit():
            qml.BasisState(np.array([0, 1, 0]), wires=range(3))
            qml.IsingXX(0.4, wires=[0, 1])
            qml.IsingYY(0.3, wires=[1, 2])
            return qml.expval(qml.Z(0))

        circuit()
        observable = 0.3 * qml.Z(1) + 0.5 * (qml.Z(0) @ qml.Z(2))
        splitter = TermsSplitter([nif_device.m_pfaffian_expval_strategy, nif_device.expval_from_probabilities_strategy])
        result = splitter(
            nif_device.state_prep_op,
            observable,
            extended_covariance_matrix=nif_device.extended_covariance_matrix,
            prob_func=nif_device.probability,
            pfaffian_chunk_size=None,
        )
        np.testing.assert_allclose(
            np.real(np.ravel(qml.math.toarray(result))),
            [float(nif_device.exact_expval(qml.Hamiltonian([0.3, 0.5], [qml.Z(1), qml.Z(0) @ qml.Z(2)])))],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_sum_with_a_non_pauli_term_is_declined_rather_than_raising(self, strategy):
        """The projector is a term of its own, which every strategy declines."""
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        assert not strategy.can_execute(state_prep_op, qml.Projector([1], wires=[0]) + 0.3 * qml.Z(1))

    def test_a_term_no_strategy_accepts_is_left_out_of_every_split(self, strategy):
        state_prep_op = qml.BasisState([0, 0], [0, 1])
        splits = strategy.split(state_prep_op, qml.Projector([1], wires=[0]) + 0.3 * qml.Z(1))
        assert [str(term.base) for term in splits[0]] == ["Z(1)"]

    def test_pennylane_numpy_coefficients_are_weighted_into_a_torch_result(self):
        """A pennylane.numpy scalar cannot multiply a torch tensor, so the coefficients are stacked."""
        from pennylane import numpy as pnp

        from matchcake import NIFDevice

        nif_device = NIFDevice(wires=3)

        @qml.qnode(nif_device)
        def circuit():
            qml.BasisState(np.array([0, 1, 0]), wires=range(3))
            qml.IsingXX(0.4, wires=[0, 1])
            return qml.expval(qml.Z(0))

        circuit()
        observable = qml.Hamiltonian(pnp.array([0.5]), [qml.Z(0)])
        splitter = TermsSplitter([nif_device.m_pfaffian_expval_strategy])
        result = splitter(
            nif_device.state_prep_op,
            observable,
            extended_covariance_matrix=nif_device.extended_covariance_matrix,
            pfaffian_chunk_size=None,
        )
        np.testing.assert_allclose(
            np.real(np.ravel(qml.math.toarray(result))),
            [float(nif_device.exact_expval(qml.Hamiltonian([0.5], [qml.Z(0)])))],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_an_observable_with_no_terms_formats_to_an_empty_hamiltonian(self, strategy):
        hamiltonian = strategy._format_observable(qml.sum(qml.Hamiltonian([], []), qml.Hamiltonian([], [])))
        assert len(hamiltonian.ops) == 0
