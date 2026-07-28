import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.probability_strategies import (
    CliffordSumStrategy,
    ExplicitSumStrategy,
    LookupTableStrategy,
    ProductStateProbabilityStrategy,
)
from matchcake.operations import CompRxRx
from matchcake.operations.state_preparation import ProductState

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)


class TestBatchedProductStateProbability:
    @staticmethod
    def selected_strategy(device):
        """Return the first strategy in the device chain that accepts the current state prep.

        :param device: Device whose ``prob_dispatcher`` chain is inspected.
        :type device: NonInteractingFermionicDevice
        :return: The strategy the dispatcher would route to.
        :rtype: matchcake.devices.probability_strategies.ProbabilityStrategy
        """
        return next(s for s in device.prob_dispatcher.strategies if s.can_execute(device.state_prep_op))

    @staticmethod
    def apply_circuit(device, prep, theta, n_wires):
        """Apply a product-state preparation followed by a CompRxRx brick-wall layer.

        :param device: Device to apply the operations on.
        :type device: NonInteractingFermionicDevice
        :param prep: State preparation operation.
        :type prep: ProductState
        :param theta: Gate angles of shape ``(n_wires - 1, 2)``.
        :type theta: torch.Tensor
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: None
        """
        device.reset()
        device.apply([prep] + [CompRxRx(theta[w], wires=[w, w + 1]) for w in range(n_wires - 1)])

    @staticmethod
    def reference_probabilities(inputs, outcomes, theta, n_wires):
        """Compute reference probabilities on ``default.qubit``, one input state at a time.

        :param inputs: Input basis states of shape ``(n_inputs, n_wires)``.
        :type inputs: np.ndarray
        :param outcomes: Queried basis outcomes of shape ``(n_outcomes, n_wires)``.
        :type outcomes: np.ndarray
        :param theta: Gate angles of shape ``(n_wires - 1, 2)``.
        :type theta: torch.Tensor
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: Probabilities of shape ``(n_outcomes, n_inputs)``.
        :rtype: np.ndarray
        """
        qubit_device = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(qubit_device, interface="torch")
        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(n_wires))
            for w in range(n_wires - 1):
                CompRxRx(theta[w], wires=[w, w + 1])
            return qml.probs(wires=range(n_wires))

        outcome_indices = [int("".join(str(int(bit)) for bit in outcome), 2) for outcome in outcomes]
        columns = [np.asarray(qml.math.detach(circuit(basis_state)))[outcome_indices] for basis_state in inputs]
        return np.stack(columns, axis=1)

    @pytest.fixture
    def circuit_data(self):
        n_wires, n_inputs, n_outcomes = 4, 5, 7
        rng = np.random.default_rng(TEST_SEED)
        inputs = (rng.random((n_inputs, n_wires)) < 0.5).astype(int)
        outcomes = (rng.random((n_outcomes, n_wires)) < 0.5).astype(int)
        theta = torch.tensor(rng.normal(size=(n_wires - 1, 2)), dtype=torch.float64, requires_grad=True)
        return n_wires, inputs, outcomes, theta

    def test_default_chain_order(self):
        device = NonInteractingFermionicDevice(wires=2)
        assert [type(s) for s in device.prob_dispatcher.strategies] == [
            LookupTableStrategy,
            ProductStateProbabilityStrategy,
            CliffordSumStrategy,
            ExplicitSumStrategy,
        ]

    def test_product_state_strategy_precedes_single_state_strategies(self):
        # Load-bearing invariant: the two trailing strategies only handle a single basis
        # state, so a batched preparation reaching them would raise. The relative order of
        # those two is inert because everything they accept is already accepted earlier.
        device = NonInteractingFermionicDevice(wires=2)
        strategy_types = [type(s) for s in device.prob_dispatcher.strategies]
        product_state_index = strategy_types.index(ProductStateProbabilityStrategy)
        assert product_state_index < strategy_types.index(ExplicitSumStrategy)
        assert product_state_index < strategy_types.index(CliffordSumStrategy)

    def test_prob_strategy_kwarg_is_ignored(self):
        # The chain is fixed; a stale ``prob_strategy=`` kwarg must not select anything.
        device = NonInteractingFermionicDevice(wires=2, prob_strategy="ExplicitSum")
        assert [type(s) for s in device.prob_dispatcher.strategies] == [
            LookupTableStrategy,
            ProductStateProbabilityStrategy,
            CliffordSumStrategy,
            ExplicitSumStrategy,
        ]

    def test_batched_prep_routes_to_product_state_strategy(self, circuit_data):
        n_wires, inputs, _, theta = circuit_data
        device = NonInteractingFermionicDevice(wires=n_wires)
        prep = ProductState.from_basis_state(inputs, wires=range(n_wires))
        self.apply_circuit(device, prep, theta, n_wires)
        assert prep.batch_size == len(inputs)
        assert isinstance(self.selected_strategy(device), ProductStateProbabilityStrategy)

    def test_unbatched_prep_routes_to_lookup_table_strategy(self, circuit_data):
        n_wires, inputs, _, theta = circuit_data
        device = NonInteractingFermionicDevice(wires=n_wires)
        prep = ProductState.from_basis_state(inputs[0], wires=range(n_wires))
        self.apply_circuit(device, prep, theta, n_wires)
        assert prep.batch_size is None
        assert isinstance(self.selected_strategy(device), LookupTableStrategy)

    def test_batched_prep_probabilities_against_qubit_device(self, circuit_data):
        n_wires, inputs, outcomes, theta = circuit_data
        device = NonInteractingFermionicDevice(wires=n_wires)
        prep = ProductState.from_basis_state(inputs, wires=range(n_wires))
        self.apply_circuit(device, prep, theta, n_wires)

        probabilities = device.get_states_probability(outcomes, device.wires)  # (n_outcomes, n_inputs)
        expected = self.reference_probabilities(inputs, outcomes, theta, n_wires)

        np.testing.assert_allclose(
            np.asarray(probabilities.detach()),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_batched_prep_probabilities_shape(self, circuit_data):
        n_wires, inputs, outcomes, theta = circuit_data
        device = NonInteractingFermionicDevice(wires=n_wires)
        prep = ProductState.from_basis_state(inputs, wires=range(n_wires))
        self.apply_circuit(device, prep, theta, n_wires)
        probabilities = device.get_states_probability(outcomes, device.wires)
        assert tuple(probabilities.shape) == (len(outcomes), len(inputs))

    def test_batched_prep_backward_pass(self, circuit_data):
        n_wires, inputs, outcomes, theta = circuit_data
        device = NonInteractingFermionicDevice(wires=n_wires)
        prep = ProductState.from_basis_state(inputs, wires=range(n_wires))
        self.apply_circuit(device, prep, theta, n_wires)

        device.get_states_probability(outcomes, device.wires).sum().backward()

        assert theta.grad is not None
        assert bool(torch.isfinite(theta.grad).all())
        assert float(torch.linalg.norm(theta.grad)) > ATOL_SCALAR_COMPARISON

    def test_batched_prep_gradient_matches_finite_differences(self):
        n_wires, n_inputs, n_outcomes = 3, 2, 2
        rng = np.random.default_rng(TEST_SEED)
        inputs = np.array([[0, 0, 0], [1, 0, 1]])[:n_inputs]
        outcomes = np.array([[0, 1, 0], [1, 1, 0]])[:n_outcomes]
        theta_values = rng.normal(size=(n_wires - 1, 2))

        def total_probability(angles):
            device = NonInteractingFermionicDevice(wires=n_wires)
            prep = ProductState.from_basis_state(inputs, wires=range(n_wires))
            self.apply_circuit(device, prep, angles, n_wires)
            return device.get_states_probability(outcomes, device.wires).sum()

        theta = torch.tensor(theta_values, dtype=torch.float64, requires_grad=True)
        total_probability(theta).backward()
        analytic_grad = np.asarray(theta.grad.detach())

        epsilon = 1e-6
        numeric_grad = np.zeros_like(theta_values)
        for index in np.ndindex(theta_values.shape):
            shifted = theta_values.copy()
            shifted[index] += epsilon
            plus = float(total_probability(torch.tensor(shifted, dtype=torch.float64)).detach())
            shifted[index] -= 2 * epsilon
            minus = float(total_probability(torch.tensor(shifted, dtype=torch.float64)).detach())
            numeric_grad[index] = (plus - minus) / (2 * epsilon)

        np.testing.assert_allclose(
            analytic_grad,
            numeric_grad,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
