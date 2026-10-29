import numpy as np
import pytest
import torch

from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.utils.covariance import (
    DEGENERACY_ATOL,
    block_diagonal_covariance,
    block_diagonalize_covariance,
)

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)


class TestCovariance:
    @staticmethod
    def random_covariance(n_modes: int, seed: int, batch_size=None) -> np.ndarray:
        """Build a random valid covariance matrix by rotating random polarizations.

        :param n_modes: Number of modes.
        :type n_modes: int
        :param seed: Seed of the random generator.
        :type seed: int
        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :return: Real antisymmetric matrix of shape ``(2 * n_modes, 2 * n_modes)`` or batched.
        :rtype: np.ndarray
        """
        rng = np.random.default_rng(seed)
        shape = (n_modes,) if batch_size is None else (batch_size, n_modes)
        polarizations = rng.uniform(-1.0, 1.0, size=shape)
        rotation = SingleParticleTransitionMatrixOperation.random_params(
            batch_size=batch_size, wires=list(range(n_modes)), seed=seed
        ).real
        block = block_diagonal_covariance(polarizations)
        return np.einsum("...ij,...ik,...kl->...jl", rotation, block, rotation)

    @staticmethod
    def reconstruct(sptm, polarizations):
        """Return ``R^T Lambda_D R`` in the backend of the inputs."""
        block = block_diagonal_covariance(polarizations)
        if isinstance(sptm, torch.Tensor):
            return torch.einsum("...ji,...jk,...kl->...il", sptm, block, sptm)
        return np.einsum("...ji,...jk,...kl->...il", sptm, block, sptm)

    @pytest.mark.parametrize("polarizations", [[1.0], [1.0, -1.0], [0.3, -0.7, 0.0], [[0.5, 0.5], [-0.2, 1.0]]])
    def test_block_diagonal_covariance_entries(self, polarizations):
        polarizations = np.asarray(polarizations)
        covariance = block_diagonal_covariance(polarizations)
        n_modes = polarizations.shape[-1]
        expected = np.zeros(polarizations.shape[:-1] + (2 * n_modes, 2 * n_modes))
        for mode in range(n_modes):
            expected[..., 2 * mode, 2 * mode + 1] = -polarizations[..., mode]
            expected[..., 2 * mode + 1, 2 * mode] = polarizations[..., mode]
        np.testing.assert_allclose(covariance, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_block_diagonal_covariance_matches_basis_state_convention(self):
        from matchcake.operations.state_preparation import ProductState

        bits = np.array([0, 1, 1, 0])
        expected = ProductState.from_basis_state(bits, wires=range(4)).covariance_matrix
        covariance = block_diagonal_covariance((-1.0) ** bits)
        np.testing.assert_allclose(
            covariance, np.asarray(expected), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_block_diagonal_covariance_preserves_backend_and_dtype(self, dtype):
        polarizations = torch.tensor([0.1, -0.4], dtype=dtype)
        covariance = block_diagonal_covariance(polarizations)
        assert isinstance(covariance, torch.Tensor)
        assert covariance.dtype == dtype
        assert isinstance(block_diagonal_covariance(np.array([0.1, -0.4])), np.ndarray)

    @pytest.mark.parametrize("n_modes, batch_size", [(1, None), (2, None), (4, None), (3, 2), (5, 3)])
    def test_block_diagonalize_covariance_reconstructs_input(self, n_modes, batch_size):
        covariance = self.random_covariance(n_modes, TEST_SEED, batch_size)
        sptm, polarizations = block_diagonalize_covariance(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("n_modes, batch_size", [(2, None), (4, None), (3, 2)])
    def test_block_diagonalize_covariance_returns_special_orthogonal_sptm(self, n_modes, batch_size):
        covariance = self.random_covariance(n_modes, TEST_SEED + 1, batch_size)
        sptm, _ = block_diagonalize_covariance(covariance)
        identity = np.broadcast_to(np.eye(2 * n_modes), sptm.shape)
        np.testing.assert_allclose(
            sptm @ np.swapaxes(sptm, -1, -2), identity, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            np.linalg.det(sptm), np.ones(sptm.shape[:-2]), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize("n_modes", [2, 3, 5])
    def test_block_diagonalize_covariance_produces_block_diagonal_form(self, n_modes):
        covariance = self.random_covariance(n_modes, TEST_SEED + 2)
        sptm, polarizations = block_diagonalize_covariance(covariance)
        rotated = sptm @ covariance @ sptm.T
        np.testing.assert_allclose(
            rotated, block_diagonal_covariance(polarizations), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_block_diagonalize_covariance_recovers_polarization_magnitudes(self):
        rng = np.random.default_rng(TEST_SEED)
        expected = rng.uniform(-1.0, 1.0, size=4)
        covariance = self.random_covariance(4, TEST_SEED + 3)
        # The random covariance was built from the polarizations drawn first with the same seed.
        rotation = SingleParticleTransitionMatrixOperation.random_params(wires=list(range(4)), seed=TEST_SEED + 3).real
        covariance = rotation.T @ block_diagonal_covariance(expected) @ rotation
        _, polarizations = block_diagonalize_covariance(covariance)
        np.testing.assert_allclose(
            np.sort(np.abs(polarizations)),
            np.sort(np.abs(expected)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "covariance",
        [
            np.zeros((4, 4)),
            block_diagonal_covariance(np.array([1.0, 1.0, 1.0])),
            block_diagonal_covariance(np.array([1.0, -1.0, 0.0, 0.0])),
            block_diagonal_covariance(np.array([0.5, 0.5, -0.5])),
        ],
    )
    def test_block_diagonalize_covariance_handles_degenerate_inputs(self, covariance):
        sptm, polarizations = block_diagonalize_covariance(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            sptm @ sptm.T, np.eye(sptm.shape[-1]), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(np.linalg.det(sptm), 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_block_diagonalize_covariance_rotated_maximally_mixed_pair(self):
        rotation = SingleParticleTransitionMatrixOperation.random_params(wires=[0, 1, 2], seed=TEST_SEED).real
        covariance = rotation.T @ block_diagonal_covariance(np.array([0.0, 0.0, 1.0])) @ rotation
        sptm, polarizations = block_diagonalize_covariance(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            np.sort(np.abs(polarizations)), [0.0, 0.0, 1.0], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_block_diagonalize_covariance_preserves_backend_and_dtype(self, dtype):
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED), dtype=dtype)
        sptm, polarizations = block_diagonalize_covariance(covariance)
        assert isinstance(sptm, torch.Tensor) and sptm.dtype == dtype
        assert isinstance(polarizations, torch.Tensor) and polarizations.dtype == dtype
        sptm_np, polarizations_np = block_diagonalize_covariance(np.asarray(covariance))
        assert isinstance(sptm_np, np.ndarray) and isinstance(polarizations_np, np.ndarray)

    def test_block_diagonalize_covariance_accepts_complex_input_with_zero_imaginary_part(self):
        covariance = self.random_covariance(2, TEST_SEED).astype(complex)
        sptm, polarizations = block_diagonalize_covariance(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm.real, polarizations.real),
            covariance.real,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_modes, batch_size", [(2, None), (3, None), (2, 2)])
    def test_block_diagonalize_covariance_gradient_of_reconstruction(self, n_modes, batch_size):
        covariance = torch.tensor(self.random_covariance(n_modes, TEST_SEED + 4, batch_size), requires_grad=True)

        def reconstruction(raw):
            antisymmetric = 0.5 * (raw - raw.mT)
            sptm, polarizations = block_diagonalize_covariance(antisymmetric)
            return self.reconstruct(sptm, polarizations)

        assert torch.autograd.gradcheck(reconstruction, (covariance,), eps=1e-6, atol=1e-5, raise_exception=True)

    def test_block_diagonalize_covariance_gradient_of_gauge_invariant_loss(self):
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED + 5), requires_grad=True)
        weight = torch.tensor(np.random.default_rng(TEST_SEED).normal(size=(6, 6)))

        def loss(raw):
            antisymmetric = 0.5 * (raw - raw.mT)
            sptm, polarizations = block_diagonalize_covariance(antisymmetric)
            evolved = sptm.mT @ block_diagonal_covariance(polarizations) @ sptm
            return torch.sum(weight * evolved) + torch.sum(polarizations**2)

        assert torch.autograd.gradcheck(loss, (covariance,), eps=1e-6, atol=1e-5, raise_exception=True)

    def test_block_diagonalize_covariance_gradient_at_pure_degenerate_point(self):
        # Every mode of a pure state has polarization +-1, a degenerate point of the decomposition. Along the
        # physical perturbations (rotations of the state) a loss on the reconstructed covariance matrix still
        # has a well-defined gradient, checked against finite differences with respect to the rotation angles.
        generator_indices = torch.triu_indices(6, 6, offset=1)
        initial = block_diagonal_covariance(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
        weight = torch.tensor(np.random.default_rng(TEST_SEED).normal(size=(6, 6)))

        def loss(angles):
            generator = torch.zeros(6, 6, dtype=angles.dtype)
            generator = generator.index_put((generator_indices[0], generator_indices[1]), angles)
            rotation = torch.linalg.matrix_exp(generator - generator.mT)
            covariance = rotation.mT @ initial @ rotation
            sptm, polarizations = block_diagonalize_covariance(covariance)
            evolved = sptm.mT @ block_diagonal_covariance(polarizations) @ sptm
            return torch.sum(weight * evolved)

        angles = torch.tensor(np.random.default_rng(TEST_SEED + 1).normal(size=15), requires_grad=True)
        assert torch.autograd.gradcheck(loss, (angles,), eps=1e-6, atol=1e-5, raise_exception=True)

    def test_block_diagonalize_covariance_gradient_projects_gauge_directions(self):
        # The polarization gradient alone must map to a block-diagonal gradient in the rotated frame.
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED + 6), requires_grad=True)
        sptm, polarizations = block_diagonalize_covariance(covariance)
        (gradient,) = torch.autograd.grad(torch.sum(polarizations * torch.arange(1.0, 4.0)), covariance)
        rotated = (sptm @ gradient @ sptm.mT).detach().numpy()
        expected = np.zeros((6, 6))
        for mode in range(3):
            expected[2 * mode, 2 * mode + 1] = -0.5 * (mode + 1)
            expected[2 * mode + 1, 2 * mode] = 0.5 * (mode + 1)
        np.testing.assert_allclose(rotated, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_degeneracy_atol_is_small(self):
        assert 0.0 < DEGENERACY_ATOL < ATOL_MATRIX_COMPARISON
