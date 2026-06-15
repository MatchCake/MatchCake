import itertools
from typing import List

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.wires import Wires

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.probability_strategies import ProductStateProbabilityStrategy
from matchcake.operations import MatchgateOperation, Rxx
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.utils import signed_pfaffian

from ...configs import ATOL_APPROX_COMPARISON, ATOL_SCALAR_COMPARISON, RTOL_APPROX_COMPARISON, set_seed


def _random_qubit_amplitudes(n_wires: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_qubit = np.zeros((n_wires, 2), dtype=complex)
    for k in range(n_wires):
        raw = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        per_qubit[k] = raw / np.linalg.norm(raw)
    return per_qubit


def _run_product_state_circuit(
    per_qubit: np.ndarray,
    n_wires: int,
    gate_seeds: List[int],
    device: qml.devices.Device,
    out_wires: List[int],
) -> np.ndarray:
    def circuit():
        ProductState(per_qubit, wires=range(n_wires))
        for idx, seed in enumerate(gate_seeds):
            MatchgateOperation.random(wires=[idx, idx + 1], seed=seed)
        return qml.probs(wires=out_wires)

    return np.asarray(qml.QNode(circuit, device)())


class TestProductStateProbabilityStrategy:
    @pytest.fixture
    def strategy(self) -> ProductStateProbabilityStrategy:
        return ProductStateProbabilityStrategy()

    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_against_default_qubit_full_probs(self, n_wires: int) -> None:
        set_seed()
        per_qubit = _random_qubit_amplitudes(n_wires, seed=42)
        gate_seeds = list(range(n_wires - 1))

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))
        ref_dev = qml.device("default.qubit", wires=range(n_wires))

        nif_probs = _run_product_state_circuit(per_qubit, n_wires, gate_seeds, nif_dev, list(range(n_wires)))
        ref_probs = _run_product_state_circuit(per_qubit, n_wires, gate_seeds, ref_dev, list(range(n_wires)))

        np.testing.assert_allclose(nif_probs, ref_probs, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_against_default_qubit_wire_marginals(self, n_wires: int) -> None:
        set_seed()
        per_qubit = _random_qubit_amplitudes(n_wires, seed=44)
        gate_seeds = [10 + idx for idx in range(n_wires - 1)]

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))
        ref_dev = qml.device("default.qubit", wires=range(n_wires))

        for wire in range(n_wires):
            nif_probs = _run_product_state_circuit(per_qubit, n_wires, gate_seeds, nif_dev, [wire])
            ref_probs = _run_product_state_circuit(per_qubit, n_wires, gate_seeds, ref_dev, [wire])
            np.testing.assert_allclose(nif_probs, ref_probs, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("n_wires", [2, 3])
    def test_against_sector_sum_oracle(self, n_wires: int) -> None:
        set_seed()
        per_qubit = _random_qubit_amplitudes(n_wires, seed=46)
        gate_seeds = [20 + idx for idx in range(n_wires - 1)]

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            for idx, seed in enumerate(gate_seeds):
                MatchgateOperation.random(wires=[idx, idx + 1], seed=seed)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        lambda_t = np.asarray(nif_dev.covariance_matrix)

        for bits in itertools.product([0, 1], repeat=n_wires):
            y = np.array(bits, dtype=int)
            expected = self._sector_sum_oracle(lambda_t, y)
            actual = float(nif_dev.get_state_probability(y))
            np.testing.assert_allclose(actual, expected, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_normalization(self, n_wires: int) -> None:
        set_seed()
        per_qubit = _random_qubit_amplitudes(n_wires, seed=48)
        gate_seeds = [30 + idx for idx in range(n_wires - 1)]

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        probs = _run_product_state_circuit(per_qubit, n_wires, gate_seeds, nif_dev, list(range(n_wires)))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("n_wires", [2, 3])
    def test_basis_state_agrees_with_lookup_table(self, n_wires: int) -> None:
        set_seed()
        gate_seeds = [50 + idx for idx in range(n_wires - 1)]
        basis_bits = np.zeros(n_wires, dtype=int)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))
        ref_dev = qml.device("default.qubit", wires=range(n_wires))

        def circuit():
            qml.BasisState(basis_bits, wires=range(n_wires))
            for idx, seed in enumerate(gate_seeds):
                MatchgateOperation.random(wires=[idx, idx + 1], seed=seed)
            return qml.probs(wires=range(n_wires))

        nif_probs = np.asarray(qml.QNode(circuit, nif_dev)())
        ref_probs = np.asarray(qml.QNode(circuit, ref_dev)())
        np.testing.assert_allclose(nif_probs, ref_probs, atol=ATOL_SCALAR_COMPARISON)

        qml.QNode(circuit, nif_dev)()
        lambda_t = nif_dev.covariance_matrix
        strat = ProductStateProbabilityStrategy()
        all_wires = nif_dev.wires

        for bits in itertools.product([0, 1], repeat=n_wires):
            y = np.array(bits, dtype=int)
            prod_prob = float(
                strat(
                    state_prep_op=nif_dev.state_prep_op,
                    target_binary_states=y,
                    wires=all_wires,
                    all_wires=all_wires,
                    covariance_matrix=lambda_t,
                )
            )
            lut_index = int("".join(map(str, bits)), 2)
            np.testing.assert_allclose(prod_prob, float(nif_probs[lut_index]), atol=ATOL_SCALAR_COMPARISON)

    def test_determinant_pfaffian_equivalence(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=60)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=7)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        lambda_t = np.asarray(nif_dev.covariance_matrix)

        for bits in itertools.product([0, 1], repeat=n_wires):
            y = np.array(bits, dtype=int)
            lambda_y = strategy.build_lambda_y(y, n_wires)
            combined = lambda_t + lambda_y

            pf_val = abs(float(signed_pfaffian(torch.as_tensor(combined, dtype=torch.float64)).item()))
            det_val = float(np.sqrt(abs(np.linalg.det(combined))))

            np.testing.assert_allclose(pf_val, det_val, atol=ATOL_SCALAR_COMPARISON)

    def test_single_call(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=62)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=9)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        lambda_t = nif_dev.covariance_matrix
        all_wires = nif_dev.wires

        all_outcomes = np.array(list(itertools.product([0, 1], repeat=n_wires)))

        single_probs = np.array(
            [
                float(
                    strategy(
                        state_prep_op=nif_dev.state_prep_op,
                        target_binary_states=y,
                        wires=all_wires,
                        all_wires=all_wires,
                        covariance_matrix=lambda_t,
                    )
                )
                for y in all_outcomes
            ]
        )

        batch_probs = np.asarray(
            strategy(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=all_outcomes,
                wires=all_wires,
                all_wires=all_wires,
                covariance_matrix=lambda_t,
            )
        )

        np.testing.assert_allclose(batch_probs, single_probs, atol=ATOL_SCALAR_COMPARISON)

    def test_call_wires_as_int(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=64)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=13)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        lambda_t = nif_dev.covariance_matrix

        prob_via_int = float(
            strategy(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=np.array([0]),
                wires=0,
                all_wires=nif_dev.wires,
                covariance_matrix=lambda_t,
            )
        )
        prob_via_list = float(
            strategy(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=np.array([0]),
                wires=[0],
                all_wires=nif_dev.wires,
                covariance_matrix=lambda_t,
            )
        )
        np.testing.assert_allclose(prob_via_int, prob_via_list, atol=ATOL_SCALAR_COMPARISON)

    def test_batch_same_wires_matches_single_calls(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=66)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=15)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        lambda_t = nif_dev.covariance_matrix
        all_wires = nif_dev.wires
        all_outcomes = np.array(list(itertools.product([0, 1], repeat=n_wires)))

        batch_probs = np.asarray(
            strategy(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=all_outcomes,
                wires=all_wires,
                all_wires=all_wires,
                covariance_matrix=lambda_t,
            )
        )
        single_probs = np.array(
            [
                float(
                    strategy(
                        state_prep_op=nif_dev.state_prep_op,
                        target_binary_states=y,
                        wires=all_wires,
                        all_wires=all_wires,
                        covariance_matrix=lambda_t,
                    )
                )
                for y in all_outcomes
            ]
        )
        np.testing.assert_allclose(batch_probs, single_probs, atol=ATOL_SCALAR_COMPARISON)

    def test_batch_different_wires_matches_single_calls(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=68)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=17)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        lambda_t = nif_dev.covariance_matrix
        all_wires = nif_dev.wires

        target_binary_states = np.array([[0], [1]])  # (2, 1)
        batch_wires = np.array([[0], [1]])  # different wire per outcome

        batch_probs = np.asarray(
            strategy(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=target_binary_states,
                wires=batch_wires,
                all_wires=all_wires,
                covariance_matrix=lambda_t,
            )
        )
        single_probs = np.array(
            [
                float(
                    strategy(
                        state_prep_op=nif_dev.state_prep_op,
                        target_binary_states=target_binary_states[i],
                        wires=Wires(batch_wires[i]),
                        all_wires=all_wires,
                        covariance_matrix=lambda_t,
                    )
                )
                for i in range(len(target_binary_states))
            ]
        )
        np.testing.assert_allclose(batch_probs, single_probs, atol=ATOL_SCALAR_COMPARISON)

    def test_batch_batched_covariance(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=70)
        all_wires = Wires(range(n_wires))

        def covariance_for(seed: int):
            nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

            def circuit():
                ProductState(per_qubit, wires=range(n_wires))
                MatchgateOperation.random(wires=[0, 1], seed=seed)
                return qml.probs(wires=range(n_wires))

            qml.QNode(circuit, nif_dev)()
            return np.asarray(nif_dev.covariance_matrix), nif_dev.state_prep_op

        # Two distinct circuits (state preparations) stacked along the covariance batch axis.
        cov_a, state_prep_op = covariance_for(19)
        cov_b, _ = covariance_for(23)
        cov_batched = np.stack([cov_a, cov_b])  # (B=2, 2n, 2n)

        # A batch of B_q outcomes queried against B state preparations must produce the full
        # (B_q, B) grid: entry [q, b] = probability of outcome q under preparation b. The
        # outcome batch and the state-prep batch are independent axes (B_q != B here).
        outcomes = np.array([[0, 0], [0, 1], [1, 1]])  # (B_q=3, k=2)

        grid_probs = np.asarray(
            strategy(
                state_prep_op=state_prep_op,
                target_binary_states=outcomes,
                wires=all_wires,
                all_wires=all_wires,
                covariance_matrix=cov_batched,
            )
        )

        ref_probs = np.array(
            [
                [
                    float(
                        strategy(
                            state_prep_op=state_prep_op,
                            target_binary_states=outcome,
                            wires=all_wires,
                            all_wires=all_wires,
                            covariance_matrix=cov,
                        )
                    )
                    for cov in (cov_a, cov_b)
                ]
                for outcome in outcomes
            ]
        )

        assert grid_probs.shape == (3, 2)
        np.testing.assert_allclose(grid_probs, ref_probs, atol=ATOL_SCALAR_COMPARISON)

    def test_build_lambda_y_single(self, strategy: ProductStateProbabilityStrategy) -> None:
        y = np.array([0, 1, 0])
        lam = strategy.build_lambda_y(y, 3)
        assert lam.shape == (6, 6)
        np.testing.assert_allclose(lam[0, 1], -1.0)  # -(-1)^0 = -1
        np.testing.assert_allclose(lam[1, 0], 1.0)
        np.testing.assert_allclose(lam[2, 3], 1.0)  # -(-1)^1 = +1
        np.testing.assert_allclose(lam[3, 2], -1.0)
        np.testing.assert_allclose(lam[4, 5], -1.0)  # -(-1)^0 = -1
        np.testing.assert_allclose(lam[5, 4], 1.0)

    def test_build_lambda_y_batch(self, strategy: ProductStateProbabilityStrategy) -> None:
        outcomes = np.array([[0, 1], [1, 0], [1, 1]])  # (3, 2)
        lam_batch = strategy.build_lambda_y(outcomes, 2)
        assert lam_batch.shape == (3, 4, 4)
        for i, y in enumerate(outcomes):
            expected = strategy.build_lambda_y(y, 2)
            np.testing.assert_allclose(lam_batch[i], expected)

    def test_extract_majorana_submatrix_batch(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 3
        per_qubit = _random_qubit_amplitudes(n_wires, seed=72)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=21)
            MatchgateOperation.random(wires=[1, 2], seed=22)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        cov = np.asarray(nif_dev.covariance_matrix)
        all_wires = nif_dev.wires

        batch_wires = np.array([[0], [1], [2]])  # (3, 1) — different wire per row
        result = ProductStateProbabilityStrategy._extract_majorana_submatrix_batch(cov, all_wires, batch_wires, k=1)
        assert result.shape == (3, 2, 2)
        for i, w in enumerate([0, 1, 2]):
            expected = ProductStateProbabilityStrategy.extract_majorana_submatrix(cov, all_wires.indices(Wires([w])))
            np.testing.assert_allclose(result[i], expected)

    def test_gradient_flow_strategy_gradcheck(self, strategy: ProductStateProbabilityStrategy) -> None:
        """Gradient flows correctly through the strategy w.r.t. the covariance matrix."""
        set_seed()
        n_wires = 2
        sp = ProductState(np.array([0.6, 0.8, 0.6, 0.8], dtype=complex), wires=[0, 1])
        all_outcomes = np.array(list(itertools.product([0, 1], repeat=n_wires)))  # (4, 2)
        all_wires = Wires(range(n_wires))

        def batch_probs(p):
            cov = p - p.t()  # real antisymmetric covariance
            return strategy(
                state_prep_op=sp,
                target_binary_states=all_outcomes,
                wires=all_wires,
                all_wires=all_wires,
                covariance_matrix=cov,
            )

        def single_prob(p):
            cov = p - p.t()
            return strategy(
                state_prep_op=sp,
                target_binary_states=np.array([0, 1]),
                wires=all_wires,
                all_wires=all_wires,
                covariance_matrix=cov,
            )

        p = torch.randn(2 * n_wires, 2 * n_wires, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(batch_probs, (p,), atol=1e-5, rtol=1e-4)
        assert torch.autograd.gradcheck(single_prob, (p,), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("init_param", [0.3, 1.1, 2.4])
    def test_gradient_through_device_matches_default_qubit(self, init_param: float) -> None:
        """End-to-end: NIF probability gradient matches default.qubit for a product-state input."""
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=80)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))
        ref_dev = qml.device("default.qubit", wires=range(n_wires))

        def make_circuit(device):
            @qml.qnode(device, interface="torch", diff_method="backprop")
            def circuit(params):
                ProductState(per_qubit, wires=range(n_wires))
                Rxx(params, wires=[0, 1])
                return qml.probs(wires=range(n_wires))

            return circuit

        nif_circuit = make_circuit(nif_dev)
        ref_circuit = make_circuit(ref_dev)

        # Self-consistency: NIF analytic gradient matches finite differences.
        gradcheck_param = torch.tensor([init_param], dtype=torch.float64).requires_grad_()
        assert torch.autograd.gradcheck(
            nif_circuit, (gradcheck_param,), eps=1e-3, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

        # Cross-check the gradient itself against the state-vector simulator. default.qubit
        # carries a leading broadcast dim, so flatten before comparing values.
        def jac(circuit):
            param = torch.tensor([init_param], dtype=torch.float64, requires_grad=True)
            return torch.autograd.functional.jacobian(lambda x: circuit(x), param).detach().ravel()

        np.testing.assert_allclose(
            np.asarray(jac(nif_circuit)),
            np.asarray(jac(ref_circuit)),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @staticmethod
    def _sector_sum_oracle(lambda_t: np.ndarray, target_binary_state: np.ndarray) -> float:
        r"""Transparent sector-sum reference for testing.

        p(y) = (1/2^n) * sum_{T subset {0..n-1}} (-1)^{|T| + sum_{k in T} y_k}
                         * Pf(Lambda(t)|_{S_T})

        where S_T = union_{k in T} {2k, 2k+1}.

        :param lambda_t: Evolved covariance matrix of shape ``(2n, 2n)``.
        :type lambda_t: np.ndarray
        :param target_binary_state: Binary outcome of shape ``(n,)``.
        :type target_binary_state: np.ndarray
        :return: Probability scalar.
        :rtype: float
        """
        n = lambda_t.shape[0] // 2
        y = np.asarray(target_binary_state, dtype=int)
        total = 0.0
        for size in range(n + 1):
            for subset in itertools.combinations(range(n), size):
                sign_exp = len(subset) + int(sum(y[k] for k in subset))
                sign = (-1) ** sign_exp
                if len(subset) == 0:
                    total += sign * 1.0
                else:
                    s_t = np.concatenate([[2 * k, 2 * k + 1] for k in subset])
                    submatrix = lambda_t[np.ix_(s_t, s_t)]
                    pf_val = float(signed_pfaffian(torch.as_tensor(submatrix, dtype=torch.float64)).item())
                    total += sign * pf_val
        return total / 2**n
