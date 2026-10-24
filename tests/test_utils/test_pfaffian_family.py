import numpy as np
import pytest
import torch

from matchcake.utils import basis_state_covariance_block, sample_outcomes, signed_pfaffian

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
)


class TestPfaffianFamily:
    @staticmethod
    def orthogonal_covariance(dim, seed):
        """Covariance of a pure Gaussian state: ``M @ M = -I`` and ``M = -M^T``."""
        generator = torch.Generator().manual_seed(seed)
        antisymmetric = torch.randn(dim, dim, dtype=torch.float64, generator=generator)
        antisymmetric = antisymmetric - antisymmetric.T
        rotation = torch.matrix_exp(antisymmetric)
        base = torch.zeros(dim, dim, dtype=torch.float64)
        for qubit in range(dim // 2):
            base[2 * qubit, 2 * qubit + 1] = 1.0
            base[2 * qubit + 1, 2 * qubit] = -1.0
        return rotation.T @ base @ rotation

    @classmethod
    def closed_form_probabilities(cls, family):
        """``p(y) = 2**-k |Pf(M + Lambda_y)|`` for every ``y``, in big-endian bit order."""
        n_qubits = family.shape[-1] // 2
        probabilities = []
        for bits in np.ndindex(*([2] * n_qubits)):
            lambda_y = basis_state_covariance_block(np.asarray(bits), 2 * n_qubits, family.dtype, family.device)
            probabilities.append(abs(float(signed_pfaffian(family + lambda_y))) * 2.0**-n_qubits)
        return np.asarray(probabilities)

    @staticmethod
    def empirical_probabilities(draws, n_qubits):
        index = (draws * (2 ** np.arange(n_qubits - 1, -1, -1))).sum(-1).ravel()
        return np.bincount(index, minlength=2**n_qubits) / index.size

    @pytest.mark.parametrize("dim, seed", [(4, 2), (6, 3), (8, 4)])
    def test_the_closed_form_probabilities_are_a_distribution(self, dim, seed):
        """Guards the oracle the sampling tests below compare against."""
        probabilities = self.closed_form_probabilities(self.orthogonal_covariance(dim, seed))
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        assert (probabilities >= -ATOL_SCALAR_COMPARISON).all()

    @pytest.mark.parametrize("dim, seed", [(4, 2), (6, 3)])
    def test_the_empirical_distribution_matches_the_pfaffian_probabilities(self, dim, seed):
        """The contract: draws are distributed as ``2**-k |Pf(M + Lambda_y)|``."""
        n_qubits = dim // 2
        family = self.orthogonal_covariance(dim, seed)
        closed_form = self.closed_form_probabilities(family)
        draws = sample_outcomes(family, 200_000, generator=torch.Generator().manual_seed(0)).numpy()
        empirical = self.empirical_probabilities(draws, n_qubits)
        # Total variation, so a single scalar bounds every outcome at once. Multinomial noise at
        # 200k shots over at most 8 outcomes sits near 6e-3, comfortably inside the tolerance.
        assert 0.5 * np.abs(empirical - closed_form).sum() < ATOL_APPROX_COMPARISON

    @pytest.mark.parametrize("bits", [(0, 0), (1, 0), (0, 1), (1, 1), (1, 0, 1)])
    def test_a_basis_state_family_is_sampled_deterministically(self, bits):
        """``Lambda_y`` puts all its mass on ``y``, so every shot must return exactly ``y``."""
        family = basis_state_covariance_block(np.asarray(bits), 2 * len(bits), torch.float64, torch.device("cpu"))
        draws = sample_outcomes(family, 64, generator=torch.Generator().manual_seed(1))
        np.testing.assert_array_equal(draws.numpy(), np.tile(np.asarray(bits), (64, 1)))

    def test_the_bit_convention_matches_basis_state_covariance_block(self):
        """Both sides encode ``y_j`` as the sign of the ``(2j, 2j+1)`` entry; a mismatch here would
        silently transpose outcomes between the sampler and the probability path."""
        for bits in np.ndindex(2, 2, 2):
            family = basis_state_covariance_block(np.asarray(bits), 6, torch.float64, torch.device("cpu"))
            draws = sample_outcomes(family, 4, generator=torch.Generator().manual_seed(2))
            np.testing.assert_array_equal(draws[0].numpy(), np.asarray(bits))

    def test_the_same_generator_seed_reproduces_the_draw(self):
        family = self.orthogonal_covariance(6, seed=5)
        first = sample_outcomes(family, 128, generator=torch.Generator().manual_seed(9))
        second = sample_outcomes(family, 128, generator=torch.Generator().manual_seed(9))
        np.testing.assert_array_equal(first.numpy(), second.numpy())

    def test_different_generator_seeds_give_different_draws(self):
        family = self.orthogonal_covariance(6, seed=5)
        first = sample_outcomes(family, 128, generator=torch.Generator().manual_seed(9))
        second = sample_outcomes(family, 128, generator=torch.Generator().manual_seed(10))
        assert not np.array_equal(first.numpy(), second.numpy())

    @pytest.mark.parametrize("shots", [1, 7, 64])
    def test_the_draw_shape_and_dtype(self, shots):
        family = self.orthogonal_covariance(6, seed=6)
        draws = sample_outcomes(family, shots, generator=torch.Generator().manual_seed(0))
        assert tuple(draws.shape) == (shots, 3)
        assert draws.dtype == torch.long
        assert set(np.unique(draws.numpy())).issubset({0, 1})

    def test_it_supports_a_batched_family(self):
        """Every batch element must be sampled from its own distribution, not from a shared one."""
        families = torch.stack([self.orthogonal_covariance(4, seed=seed) for seed in (11, 12, 13)])
        draws = sample_outcomes(families, 100_000, generator=torch.Generator().manual_seed(3))
        assert tuple(draws.shape) == (100_000, 3, 2)
        for element in range(3):
            closed_form = self.closed_form_probabilities(families[element])
            empirical = self.empirical_probabilities(draws[:, element, :].numpy(), 2)
            assert 0.5 * np.abs(empirical - closed_form).sum() < ATOL_APPROX_COMPARISON

    def test_it_supports_a_single_qubit_family(self):
        """``k = 1`` never enters the Schur-elimination branch, so it is its own path."""
        family = torch.tensor([[0.0, 0.6], [-0.6, 0.0]], dtype=torch.float64)
        draws = sample_outcomes(family, 200_000, generator=torch.Generator().manual_seed(4))
        assert tuple(draws.shape) == (200_000, 1)
        np.testing.assert_allclose(draws.double().mean().item(), 0.8, atol=ATOL_APPROX_COMPARISON)

    def test_it_tolerates_a_zero_pivot(self):
        """A vanishing pivot must not divide by zero: the level is unbiased and the sweep continues."""
        family = torch.zeros(4, 4, dtype=torch.float64)
        family[2, 3], family[3, 2] = 1.0, -1.0
        draws = sample_outcomes(family, 20_000, generator=torch.Generator().manual_seed(5))
        assert torch.isfinite(draws.double()).all()
        np.testing.assert_allclose(draws[:, 0].double().mean().item(), 0.5, atol=ATOL_APPROX_COMPARISON)
        np.testing.assert_array_equal(draws[:, 1].numpy(), np.ones(20_000, dtype=np.int64))

    def test_it_refuses_a_complex_family(self):
        family = self.orthogonal_covariance(4, seed=7).to(torch.complex128)
        with pytest.raises(ValueError, match="real covariance family"):
            sample_outcomes(family, 1)

    @pytest.mark.parametrize("shape", [(3, 3), (4,), (2, 4, 3), (5, 5)])
    def test_it_refuses_a_family_that_is_not_a_batch_of_even_square_matrices(self, shape):
        with pytest.raises(ValueError, match=r"expected \(\.\.\., 2k, 2k\)"):
            sample_outcomes(torch.zeros(*shape, dtype=torch.float64), 1)

    @pytest.mark.parametrize("shots", [0, -1])
    def test_it_refuses_a_non_positive_shot_count(self, shots):
        with pytest.raises(ValueError, match="shots must be >= 1"):
            sample_outcomes(self.orthogonal_covariance(4, seed=8), shots)

    def test_it_does_not_mutate_its_input(self):
        family = self.orthogonal_covariance(6, seed=14)
        before = family.clone()
        sample_outcomes(family, 32, generator=torch.Generator().manual_seed(0))
        np.testing.assert_array_equal(family.numpy(), before.numpy())

    def test_the_sampler_runs_under_no_grad(self):
        """The draw is non-differentiable by construction, so the k-level Schur elimination must
        not build an autograd graph for a family that carries one."""
        assert hasattr(sample_outcomes, "__wrapped__"), "sample_outcomes lost its @torch.no_grad()"
        family = self.orthogonal_covariance(6, seed=15).requires_grad_(True)
        with torch.enable_grad():
            draws = sample_outcomes(family, 8, generator=torch.Generator().manual_seed(0))
        assert not draws.requires_grad
        assert draws.grad_fn is None
