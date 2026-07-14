import numpy as np
import pytest
import torch

from matchcake import utils
from matchcake.devices import NIFDevice
from matchcake.devices.probability_strategies.product_state_strategy import ProductStateProbabilityStrategy
from matchcake.utils._pfaffian_family import (
    OutcomeFamily,
    build_lambda_y,
    outcome_bits,
    sample_outcomes,
)
from tests.configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)


class TestPfaffianFamily:
    @staticmethod
    def _random_skew(rng: np.random.Generator, m: int, is_complex: bool = False) -> torch.Tensor:
        x = rng.standard_normal((m, m))
        if is_complex:
            x = x + 1j * rng.standard_normal((m, m))
        x = x - x.T
        return torch.from_numpy(x)

    @staticmethod
    def _random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
        q, r = np.linalg.qr(rng.standard_normal((n, n)))
        return q * np.sign(np.diag(r))

    @staticmethod
    def _computational_basis_covariance(bits: np.ndarray) -> torch.Tensor:
        k = len(bits)
        signs = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
        lam = torch.zeros(2 * k, 2 * k, dtype=torch.float64)
        block = torch.arange(k)
        lam[2 * block, 2 * block + 1] = torch.from_numpy(signs)
        lam[2 * block + 1, 2 * block] = -torch.from_numpy(signs)
        return lam

    @classmethod
    def _physical_covariance(cls, rng: np.random.Generator, k: int, bits: np.ndarray = None) -> torch.Tensor:
        if bits is None:
            bits = np.zeros(k, dtype=np.int64)
        lam = cls._computational_basis_covariance(np.asarray(bits)).numpy()
        q = cls._random_orthogonal(rng, 2 * k)
        return torch.from_numpy(q.T @ lam @ q)

    @classmethod
    def _mixed_covariance(cls, rng: np.random.Generator, k: int, purity: float = 0.5) -> torch.Tensor:
        blocks = torch.zeros(2 * k, 2 * k, dtype=torch.float64)
        idx = torch.arange(k)
        blocks[2 * idx, 2 * idx + 1] = purity
        blocks[2 * idx + 1, 2 * idx] = -purity
        q = torch.from_numpy(cls._random_orthogonal(rng, 2 * k))
        return q.T @ blocks @ q

    @classmethod
    def _hermitian_pair_grid(cls, rng: np.random.Generator, chi: int, m: int) -> torch.Tensor:
        grid = np.zeros((chi, chi, m, m), dtype=np.complex128)
        for a in range(chi):
            grid[a, a] = cls._random_skew(rng, m).numpy().astype(np.complex128)
            for b in range(a + 1, chi):
                gab = cls._random_skew(rng, m, is_complex=True).numpy()
                grid[a, b] = gab
                grid[b, a] = np.conj(gab)
        return torch.from_numpy(grid)

    @classmethod
    def _brute_force_family(cls, m_fixed: torch.Tensor) -> torch.Tensor:
        k = m_fixed.shape[-1] // 2
        bits = outcome_bits(k)
        lam = build_lambda_y(bits, dtype=m_fixed.dtype)
        values = [utils.pfaffian(m_fixed + lam[i], sign=True) for i in range(2**k)]
        return torch.stack([torch.as_tensor(v, dtype=m_fixed.dtype) for v in values])

    def test_outcome_bits_convention(self):
        bits = outcome_bits(3)
        assert bits.shape == (8, 3)
        assert bits[0].tolist() == [0, 0, 0]
        assert bits[1].tolist() == [0, 0, 1]
        assert bits[4].tolist() == [1, 0, 0]

    def test_build_lambda_y_matches_expected_blocks(self):
        bits = torch.tensor([[1, 0]])
        lam = build_lambda_y(bits, dtype=torch.float64)[0]
        assert lam.shape == (4, 4)
        assert lam[0, 1].item() == pytest.approx(1.0)
        assert lam[2, 3].item() == pytest.approx(-1.0)
        torch.testing.assert_close(lam, -lam.T)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_tree_matches_brute_force_real(self, k):
        rng = np.random.default_rng(TEST_SEED + k)
        m_fixed = self._random_skew(rng, 2 * k)
        expected = self._brute_force_family(m_fixed)
        result = OutcomeFamily(m_fixed).all_pfaffians()
        torch.testing.assert_close(result, expected, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_tree_matches_brute_force_complex(self, k):
        rng = np.random.default_rng(TEST_SEED + k)
        m_fixed = self._random_skew(rng, 2 * k, is_complex=True)
        expected = self._brute_force_family(m_fixed)
        result = OutcomeFamily(m_fixed).all_pfaffians()
        torch.testing.assert_close(result, expected, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    def test_tree_batched_matches_per_element(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = torch.stack([torch.stack([self._random_skew(rng, 6) for _ in range(3)]) for _ in range(2)])
        batched = OutcomeFamily(batch).all_pfaffians()
        assert batched.shape == (2, 3, 8)
        for i in range(2):
            for j in range(3):
                single = OutcomeFamily(batch[i, j]).all_pfaffians()
                torch.testing.assert_close(
                    batched[i, j], single, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON
                )

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_probabilities_normalize_on_physical_covariance(self, k):
        rng = np.random.default_rng(TEST_SEED + k)
        m_fixed = self._physical_covariance(rng, k)
        probs = OutcomeFamily(m_fixed).all_probabilities()
        assert probs.shape == (2**k,)
        assert torch.all(probs >= 0)
        assert probs.sum().item() == pytest.approx(1.0, abs=ATOL_SCALAR_COMPARISON)

    def test_deterministic_state_prunes_to_delta(self):
        bits = np.array([1, 0, 1])
        m_fixed = self._computational_basis_covariance(bits)
        probs = OutcomeFamily(m_fixed).all_probabilities()
        expected_index = int("".join(map(str, bits)), 2)
        assert probs[expected_index].item() == pytest.approx(1.0, abs=ATOL_SCALAR_COMPARISON)
        others = torch.cat([probs[:expected_index], probs[expected_index + 1 :]])
        assert torch.all(others == 0.0)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_pure_state_parity_conservation(self, k):
        rng = np.random.default_rng(TEST_SEED + k)
        m_fixed = self._physical_covariance(rng, k)
        probs = OutcomeFamily(m_fixed).all_probabilities()
        state_parity = round(float(utils.pfaffian(m_fixed, sign=True).real)) * (-1) ** k
        outcome_parity = 1 - 2 * (outcome_bits(k).sum(-1) % 2)
        forbidden = probs[outcome_parity != state_parity]
        allowed = probs[outcome_parity == state_parity]
        assert torch.all(forbidden < ATOL_MATRIX_COMPARISON)
        assert allowed.sum().item() == pytest.approx(1.0, abs=ATOL_SCALAR_COMPARISON)

    def test_hermitian_pair_grid_symmetry(self):
        rng = np.random.default_rng(TEST_SEED)
        grid = self._hermitian_pair_grid(rng, chi=3, m=4)
        values = OutcomeFamily(grid).all_pfaffians()
        torch.testing.assert_close(
            values, values.transpose(0, 1).conj(), rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON
        )
        assert torch.all(values[torch.arange(3), torch.arange(3)].imag.abs() < ATOL_MATRIX_COMPARISON)

    def test_gradcheck_through_tree(self):
        torch.manual_seed(TEST_SEED)
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: OutcomeFamily(t - t.transpose(-1, -2)).all_pfaffians(), (x,))

    def test_gradcheck_probabilities_near_physical(self):
        rng = np.random.default_rng(TEST_SEED)
        noise = torch.from_numpy(rng.standard_normal((4, 4)))
        base = self._physical_covariance(rng, 2) + 0.1 * (noise - noise.T)
        base = base.requires_grad_(True)
        skew = lambda t: 0.5 * (t - t.transpose(-1, -2))  # noqa: E731
        assert torch.autograd.gradcheck(lambda t: OutcomeFamily(skew(t)).all_probabilities(), (base,))

    def test_work_dtype_promotion(self):
        rng = np.random.default_rng(TEST_SEED)
        m32 = self._random_skew(rng, 6).to(torch.float32)
        result = OutcomeFamily(m32, work_dtype=torch.float64).all_pfaffians()
        assert result.dtype == torch.float64
        expected = self._brute_force_family(m32.to(torch.float64))
        torch.testing.assert_close(result, expected, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    def test_n_outcomes_property(self):
        rng = np.random.default_rng(TEST_SEED)
        assert OutcomeFamily(self._random_skew(rng, 6)).n_outcomes == 8

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="even"):
            OutcomeFamily(torch.zeros(3, 3))
        with pytest.raises(ValueError, match="2k"):
            OutcomeFamily(torch.zeros(4, 3))
        with pytest.raises(ValueError, match="prune_threshold"):
            OutcomeFamily(torch.zeros(4, 4), prune_threshold=-1.0)
        with pytest.raises(ValueError, match="real"):
            OutcomeFamily(torch.zeros(4, 4, dtype=torch.complex128)).all_probabilities()

    def test_bit_order_matches_production_enumeration(self):
        rng = np.random.default_rng(TEST_SEED + 7)
        for k in [2, 3, 4]:
            m_fixed = self._physical_covariance(rng, k)
            new_probs = OutcomeFamily(m_fixed).all_probabilities()
            target_arr = NIFDevice.states_to_binary(np.arange(2**k), k)
            lambda_y = torch.as_tensor(
                ProductStateProbabilityStrategy.build_lambda_y(target_arr, k), dtype=m_fixed.dtype
            )
            combined = m_fixed.unsqueeze(0) + lambda_y
            old_probs = (2.0**-k) * torch.real(utils.pfaffian(combined, sign=False))
            torch.testing.assert_close(new_probs, old_probs, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    def test_bit_order_signed_generic_matrix(self):
        rng = np.random.default_rng(TEST_SEED + 11)
        for k in [2, 3, 4]:
            m_fixed = self._random_skew(rng, 2 * k)
            new_pf = OutcomeFamily(m_fixed).all_pfaffians()
            target_arr = NIFDevice.states_to_binary(np.arange(2**k), k)
            lambda_y = torch.as_tensor(
                ProductStateProbabilityStrategy.build_lambda_y(target_arr, k), dtype=m_fixed.dtype
            )
            combined = m_fixed.unsqueeze(0) + lambda_y
            old_pf = utils.pfaffian(combined, sign=True)
            torch.testing.assert_close(new_pf, old_pf, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    def test_gradcheck_complex_through_tree(self):
        torch.manual_seed(TEST_SEED)
        x = torch.randn(4, 4, dtype=torch.complex128, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: OutcomeFamily(t - t.transpose(-1, -2)).all_pfaffians(), (x,))

    def test_mixed_state_distribution_and_sampler(self):
        rng = np.random.default_rng(TEST_SEED + 5)
        k = 3
        m_fixed = self._mixed_covariance(rng, k, purity=0.5)
        probs = OutcomeFamily(m_fixed).all_probabilities()
        assert torch.all(probs > ATOL_MATRIX_COMPARISON)
        assert probs.sum().item() == pytest.approx(1.0, abs=ATOL_SCALAR_COMPARISON)
        expected = self._brute_force_family(m_fixed).abs() * (2.0**-k)
        torch.testing.assert_close(probs, expected, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)
        samples = sample_outcomes(m_fixed, shots=40_000, generator=torch.Generator().manual_seed(TEST_SEED))
        weights = 2 ** torch.arange(k - 1, -1, -1)
        indices = (samples * weights).sum(-1)
        empirical = torch.bincount(indices, minlength=2**k).double() / 40_000
        assert 0.5 * (empirical - probs).abs().sum().item() < 0.05

    @staticmethod
    def _exact_zero_pivot_matrix(dtype: torch.dtype) -> torch.Tensor:
        # M[0,1]=1 forces the y_0=0 pivot m[0,1] + (2*0 - 1) = 0 exactly, so the unpivoted sweep
        # prunes outcomes 0 and 1 to spurious zeros while their full Pfaffians are nonzero.
        entries = {(0, 1): 1.0, (0, 2): 0.7, (0, 3): -0.4, (1, 2): 0.9, (1, 3): 0.3, (2, 3): 0.5}
        m_fixed = torch.zeros(4, 4, dtype=dtype)
        for (i, j), value in entries.items():
            m_fixed[i, j] = value
            m_fixed[j, i] = -value
        if dtype.is_complex:
            m_fixed[0, 2] = m_fixed[0, 2] + 0.5j
            m_fixed[2, 0] = m_fixed[2, 0] - 0.5j
        return m_fixed

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_exact_zero_pivot_fallback_repairs_spurious_zero(self, dtype):
        m_fixed = self._exact_zero_pivot_matrix(dtype)
        lam = build_lambda_y(outcome_bits(2), dtype=dtype)
        oracle = torch.stack([utils.pfaffian(m_fixed + lam[i], sign=True) for i in range(4)])
        default = OutcomeFamily(m_fixed).all_pfaffians()
        fixed = OutcomeFamily(m_fixed).all_pfaffians(zero_pivot_fallback=True)
        assert default[0].item() == 0.0 and default[1].item() == 0.0
        torch.testing.assert_close(fixed, oracle, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    def test_zero_pivot_fallback_no_op_without_dead_lanes(self):
        rng = np.random.default_rng(TEST_SEED)
        m_fixed = self._random_skew(rng, 6)
        without = OutcomeFamily(m_fixed).all_pfaffians()
        with_fallback = OutcomeFamily(m_fixed).all_pfaffians(zero_pivot_fallback=True)
        torch.testing.assert_close(without, with_fallback, rtol=RTOL_MATRIX_COMPARISON, atol=ATOL_MATRIX_COMPARISON)

    def test_zero_pivot_fallback_gradient_finite_at_dead_lane(self):
        x = self._exact_zero_pivot_matrix(torch.float64).clone().requires_grad_(True)
        pfaffians = OutcomeFamily(x).all_pfaffians(zero_pivot_fallback=True)
        pfaffians.abs().sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_gradcheck_zero_pivot_fallback(self):
        torch.manual_seed(TEST_SEED)
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda t: OutcomeFamily(t - t.transpose(-1, -2)).all_pfaffians(zero_pivot_fallback=True), (x,)
        )

    def test_sampler_deterministic_state_samples_exactly(self):
        bits = np.array([1, 0, 1, 1])
        m_fixed = self._computational_basis_covariance(bits)
        samples = sample_outcomes(m_fixed, shots=64, generator=torch.Generator().manual_seed(TEST_SEED))
        assert samples.shape == (64, 4)
        assert torch.all(samples == torch.from_numpy(bits).long())

    def test_sampler_matches_distribution(self):
        rng = np.random.default_rng(TEST_SEED)
        k, shots = 5, 40_000
        m_fixed = self._physical_covariance(rng, k)
        probs = OutcomeFamily(m_fixed).all_probabilities()
        samples = sample_outcomes(m_fixed, shots=shots, generator=torch.Generator().manual_seed(TEST_SEED))
        weights = 2 ** torch.arange(k - 1, -1, -1)
        indices = (samples * weights).sum(-1)
        counts = torch.bincount(indices, minlength=2**k).double()
        empirical = counts / shots
        total_variation = 0.5 * (empirical - probs).abs().sum().item()
        assert total_variation < 0.05

    def test_sampler_agrees_with_first_bit_marginal(self):
        rng = np.random.default_rng(TEST_SEED + 1)
        m_fixed = self._physical_covariance(rng, 4)
        p_one = (m_fixed[0, 1].clamp(-1.0, 1.0) + 1.0) / 2.0
        samples = sample_outcomes(m_fixed, shots=50_000, generator=torch.Generator().manual_seed(TEST_SEED))
        freq = samples[:, 0].double().mean()
        assert freq.item() == pytest.approx(p_one.item(), abs=0.01)

    def test_sampler_batched_shapes(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = torch.stack([self._physical_covariance(rng, 3) for _ in range(5)])
        samples = sample_outcomes(batch, shots=7, generator=torch.Generator().manual_seed(TEST_SEED))
        assert samples.shape == (7, 5, 3)
        assert set(samples.unique().tolist()) <= {0, 1}

    def test_sampler_validation(self):
        with pytest.raises(ValueError, match="real"):
            sample_outcomes(torch.zeros(4, 4, dtype=torch.complex128), shots=1)
        with pytest.raises(ValueError, match="shots"):
            sample_outcomes(torch.zeros(4, 4), shots=0)
        with pytest.raises(ValueError, match="2k"):
            sample_outcomes(torch.zeros(4, 3), shots=1)
