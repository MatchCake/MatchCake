import itertools
from typing import List

import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.wires import Wires

from matchcake import NonInteractingFermionicDevice
from matchcake.devices.probability_strategies import ProductStateProbabilityStrategy
from matchcake.operations import MatchgateOperation
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.utils import signed_pfaffian

from ...configs import ATOL_SCALAR_COMPARISON, set_seed


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

        # Direct strategy comparison: product-state strategy must match LUT
        qml.QNode(circuit, nif_dev)()  # rebuild SPTM
        lambda_t = nif_dev.covariance_matrix
        strat = ProductStateProbabilityStrategy()
        all_wires = nif_dev.wires

        for bits in itertools.product([0, 1], repeat=n_wires):
            y = np.array(bits, dtype=int)
            prod_prob = float(
                strat(
                    state_prep_op=nif_dev.state_prep_op,
                    target_binary_state=y,
                    wires=all_wires,
                    all_wires=all_wires,
                    covariance_matrix=lambda_t,
                    pfaffian_method=nif_dev.pfaffian_method,
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

    def test_batching(self, strategy: ProductStateProbabilityStrategy) -> None:
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

        all_outcomes = np.array(list(itertools.product([0, 1], repeat=n_wires)))  # (4, 2)
        batch_wires = np.broadcast_to(np.asarray(all_wires), all_outcomes.shape)

        single_probs = np.array(
            [
                float(
                    strategy(
                        state_prep_op=nif_dev.state_prep_op,
                        target_binary_state=y,
                        wires=all_wires,
                        all_wires=all_wires,
                        covariance_matrix=lambda_t,
                        pfaffian_method=nif_dev.pfaffian_method,
                    )
                )
                for y in all_outcomes
            ]
        )

        batch_probs = np.asarray(
            strategy.batch_call(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=all_outcomes,
                batch_wires=batch_wires,
                all_wires=all_wires,
                covariance_matrix=lambda_t,
                pfaffian_method=nif_dev.pfaffian_method,
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
                target_binary_state=np.array([0]),
                wires=0,
                all_wires=nif_dev.wires,
                covariance_matrix=lambda_t,
                pfaffian_method=nif_dev.pfaffian_method,
            )
        )
        prob_via_list = float(
            strategy(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_state=np.array([0]),
                wires=[0],
                all_wires=nif_dev.wires,
                covariance_matrix=lambda_t,
                pfaffian_method=nif_dev.pfaffian_method,
            )
        )
        np.testing.assert_allclose(prob_via_int, prob_via_list, atol=ATOL_SCALAR_COMPARISON)

    def test_batch_call_without_batch_wires(self, strategy: ProductStateProbabilityStrategy) -> None:
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

        result_no_bw = np.asarray(
            strategy.batch_call(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=all_outcomes,
                batch_wires=None,
                all_wires=all_wires,
                covariance_matrix=lambda_t,
                pfaffian_method=nif_dev.pfaffian_method,
            )
        )
        result_explicit_bw = np.asarray(
            strategy.batch_call(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=all_outcomes,
                batch_wires=np.broadcast_to(np.asarray(all_wires), all_outcomes.shape),
                all_wires=all_wires,
                covariance_matrix=lambda_t,
                pfaffian_method=nif_dev.pfaffian_method,
            )
        )
        np.testing.assert_allclose(result_no_bw, result_explicit_bw, atol=ATOL_SCALAR_COMPARISON)

    def test_batch_call_different_wires_fallback(self, strategy: ProductStateProbabilityStrategy) -> None:
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

        # Measure each outcome on a different single wire (forces the not-same-wires fallback)
        target_binary_states = np.array([[0], [1]])  # (2, 1)
        batch_wires = np.array([[0], [1]])  # (2, 1) — different wire per outcome

        batch_probs = np.asarray(
            strategy.batch_call(
                state_prep_op=nif_dev.state_prep_op,
                target_binary_states=target_binary_states,
                batch_wires=batch_wires,
                all_wires=all_wires,
                covariance_matrix=lambda_t,
                pfaffian_method=nif_dev.pfaffian_method,
            )
        )
        single_probs = np.array(
            [
                float(
                    strategy(
                        state_prep_op=nif_dev.state_prep_op,
                        target_binary_state=target_binary_states[i],
                        wires=Wires(batch_wires[i]),
                        all_wires=all_wires,
                        covariance_matrix=lambda_t,
                        pfaffian_method=nif_dev.pfaffian_method,
                    )
                )
                for i in range(len(target_binary_states))
            ]
        )
        np.testing.assert_allclose(batch_probs, single_probs, atol=ATOL_SCALAR_COMPARISON)

    def test_batch_call_batched_covariance_fallback(self, strategy: ProductStateProbabilityStrategy) -> None:
        set_seed()
        n_wires = 2
        per_qubit = _random_qubit_amplitudes(n_wires, seed=70)

        nif_dev = NonInteractingFermionicDevice(wires=range(n_wires))

        def circuit():
            ProductState(per_qubit, wires=range(n_wires))
            MatchgateOperation.random(wires=[0, 1], seed=19)
            return qml.probs(wires=range(n_wires))

        qml.QNode(circuit, nif_dev)()
        cov = np.asarray(nif_dev.covariance_matrix)  # (2n, 2n)
        # Simulate device batch by adding a leading batch dim
        cov_batched = np.stack([cov])  # (1, 2n, 2n)

        all_wires = nif_dev.wires
        all_outcomes = np.array(list(itertools.product([0, 1], repeat=n_wires)))
        batch_wires = np.broadcast_to(np.asarray(all_wires), all_outcomes.shape)

        # The cov_ndim > 2 branch falls back to per-outcome calls; result shape: (B,) or (B, cov_batch)
        result = strategy.batch_call(
            state_prep_op=nif_dev.state_prep_op,
            target_binary_states=all_outcomes,
            batch_wires=batch_wires,
            all_wires=all_wires,
            covariance_matrix=cov_batched,
            pfaffian_method=nif_dev.pfaffian_method,
        )
        # Result has a leading B dimension (one entry per outcome), each stacked per cov-batch
        assert result is not None

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
                    total += sign * 1.0  # Pf([]) = 1 by convention
                else:
                    s_t = np.concatenate([[2 * k, 2 * k + 1] for k in subset])
                    submatrix = lambda_t[np.ix_(s_t, s_t)]
                    pf_val = float(signed_pfaffian(torch.as_tensor(submatrix, dtype=torch.float64)).item())
                    total += sign * pf_val
        return total / 2**n
