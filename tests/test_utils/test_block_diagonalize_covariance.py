import numpy as np
import pytest
import torch

from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils.block_diagonalize_covariance import (
    DEGENERACY_ATOL,
    BlockDiagonalizeCovariance,
)

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)


class TestBlockDiagonalizeCovariance:
    @staticmethod
    def block_diagonal(polarizations: torch.Tensor) -> torch.Tensor:
        """Build the block-diagonal matrix with ``-polarizations`` on the upper off-diagonal of each mode.

        :param polarizations: Polarizations of shape ``(..., m)``.
        :type polarizations: torch.Tensor
        :return: Antisymmetric matrix of shape ``(..., 2m, 2m)``.
        :rtype: torch.Tensor
        """
        mode_block = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=polarizations.dtype)
        mode_identity = torch.eye(polarizations.shape[-1], dtype=polarizations.dtype)
        blocks = torch.einsum("...j,jk,ab->...jakb", polarizations, mode_identity, mode_block)
        size = 2 * polarizations.shape[-1]
        return blocks.reshape(*polarizations.shape[:-1], size, size)

    @staticmethod
    def random_covariance(n_modes: int, seed: int, batch_shape=()) -> torch.Tensor:
        """Build random covariance matrices by rotating random polarizations.

        :param n_modes: Number of modes.
        :type n_modes: int
        :param seed: Seed of the random generator.
        :type seed: int
        :param batch_shape: Leading batch shape.
        :type batch_shape: tuple
        :return: Real antisymmetric tensor of shape ``batch_shape + (2 * n_modes, 2 * n_modes)``.
        :rtype: torch.Tensor
        """
        rng = np.random.default_rng(seed)
        polarizations = torch.tensor(rng.uniform(-1.0, 1.0, size=tuple(batch_shape) + (n_modes,)))
        rotation = torch.tensor(
            SingleParticleTransitionMatrixOperation.random_params(wires=list(range(n_modes)), seed=seed).real
        )
        block = TestBlockDiagonalizeCovariance.block_diagonal(polarizations)
        return torch.einsum("ij,...ik,kl->...jl", rotation, block, rotation)

    @staticmethod
    def reconstruct(sptm: torch.Tensor, polarizations: torch.Tensor) -> torch.Tensor:
        """Return ``R^T Lambda_D R``.

        :param sptm: Transition matrices of shape ``(..., 2m, 2m)``.
        :type sptm: torch.Tensor
        :param polarizations: Polarizations of shape ``(..., m)``.
        :type polarizations: torch.Tensor
        :return: The reconstructed matrices.
        :rtype: torch.Tensor
        """
        return sptm.mT @ TestBlockDiagonalizeCovariance.block_diagonal(polarizations) @ sptm

    def test_degeneracy_atol_is_small(self):
        assert 0.0 < DEGENERACY_ATOL < ATOL_MATRIX_COMPARISON

    @pytest.mark.parametrize("n_modes, batch_shape", [(1, ()), (2, ()), (4, ()), (3, (2,)), (2, (2, 3))])
    def test_forward_reconstructs_input(self, n_modes, batch_shape):
        covariance = self.random_covariance(n_modes, TEST_SEED, batch_shape)
        sptm, polarizations = BlockDiagonalizeCovariance.apply(covariance)
        assert tuple(sptm.shape) == tuple(batch_shape) + (2 * n_modes, 2 * n_modes)
        assert tuple(polarizations.shape) == tuple(batch_shape) + (n_modes,)
        torch.testing.assert_close(
            self.reconstruct(sptm, polarizations),
            covariance,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_modes, batch_shape", [(2, ()), (4, ()), (3, (2,))])
    def test_forward_returns_special_orthogonal_matrices(self, n_modes, batch_shape):
        covariance = self.random_covariance(n_modes, TEST_SEED + 1, batch_shape)
        sptm, _ = BlockDiagonalizeCovariance.apply(covariance)
        identity = torch.eye(2 * n_modes, dtype=sptm.dtype).expand(sptm.shape)
        torch.testing.assert_close(sptm @ sptm.mT, identity, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        torch.testing.assert_close(
            torch.linalg.det(sptm),
            torch.ones(tuple(batch_shape), dtype=sptm.dtype),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("polarizations", [[0.0, 0.0], [1.0, 1.0, 1.0], [1.0, -1.0, 0.0, 0.0], [0.5, 0.5, -0.5]])
    def test_forward_handles_degenerate_inputs(self, polarizations):
        covariance = self.block_diagonal(torch.tensor(polarizations, dtype=torch.float64))
        sptm, recovered = BlockDiagonalizeCovariance.apply(covariance)
        torch.testing.assert_close(
            self.reconstruct(sptm, recovered), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            np.sort(np.abs(recovered.numpy())),
            np.sort(np.abs(polarizations)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_forward_antisymmetrizes_its_input(self):
        covariance = self.random_covariance(2, TEST_SEED + 2)
        symmetric_part = torch.ones_like(covariance)
        sptm, polarizations = BlockDiagonalizeCovariance.apply(covariance + symmetric_part)
        torch.testing.assert_close(
            self.reconstruct(sptm, polarizations),
            covariance,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_keeps_precision(self, dtype):
        covariance = self.random_covariance(3, TEST_SEED).to(dtype)
        sptm, polarizations = BlockDiagonalizeCovariance.apply(covariance)
        assert sptm.dtype == dtype
        assert polarizations.dtype == dtype

    @pytest.mark.parametrize("n_modes, batch_shape", [(2, ()), (3, ()), (2, (2,))])
    def test_backward_of_reconstruction(self, n_modes, batch_shape):
        covariance = self.random_covariance(n_modes, TEST_SEED + 3, batch_shape).requires_grad_(True)

        def reconstruction(raw):
            sptm, polarizations = BlockDiagonalizeCovariance.apply(0.5 * (raw - raw.mT))
            return self.reconstruct(sptm, polarizations)

        assert torch.autograd.gradcheck(
            reconstruction, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_backward_of_polarizations(self):
        covariance = self.random_covariance(3, TEST_SEED + 4).requires_grad_(True)

        def squared_polarizations(raw):
            _, polarizations = BlockDiagonalizeCovariance.apply(0.5 * (raw - raw.mT))
            return torch.sum(polarizations**2)

        assert torch.autograd.gradcheck(
            squared_polarizations, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_backward_projects_gauge_directions(self):
        covariance = self.random_covariance(3, TEST_SEED + 5).requires_grad_(True)
        sptm, polarizations = BlockDiagonalizeCovariance.apply(covariance)
        weights = torch.arange(1.0, 4.0, dtype=polarizations.dtype)
        (gradient,) = torch.autograd.grad(torch.sum(polarizations * weights), covariance)
        rotated = sptm.detach() @ gradient @ sptm.detach().mT
        torch.testing.assert_close(
            rotated, 0.5 * self.block_diagonal(weights), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_backward_is_finite_at_degenerate_point(self):
        covariance = self.block_diagonal(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)).requires_grad_(True)
        sptm, polarizations = BlockDiagonalizeCovariance.apply(covariance)
        weight = torch.tensor(np.random.default_rng(TEST_SEED).normal(size=(6, 6)))
        (gradient,) = torch.autograd.grad(torch.sum(weight * sptm) + torch.sum(polarizations), covariance)
        assert torch.all(torch.isfinite(gradient))
        torch.testing.assert_close(gradient, -gradient.mT, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
