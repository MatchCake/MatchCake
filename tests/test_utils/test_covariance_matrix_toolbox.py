import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.operations.state_preparation import ProductState
from matchcake.utils.covariance_matrix_toolbox import CovarianceMatrixToolbox
from matchcake.utils.majorana import get_majorana

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)


class TestCovarianceMatrixToolbox:
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
        block = np.asarray(CovarianceMatrixToolbox.from_polarizations(polarizations))
        return np.einsum("...ij,...ik,...kl->...jl", rotation, block, rotation)

    @staticmethod
    def reconstruct(sptm, polarizations):
        """Return ``R^T Lambda_D R`` in the backend of the inputs.

        :param sptm: Transition matrix of shape ``(..., 2m, 2m)``.
        :param polarizations: Polarizations of shape ``(..., m)``.
        :return: The reconstructed covariance matrix.
        """
        block = CovarianceMatrixToolbox.from_polarizations(polarizations)
        if isinstance(sptm, torch.Tensor):
            return torch.einsum("...ji,...jk,...kl->...il", sptm, block, sptm)
        return np.einsum("...ji,...jk,...kl->...il", sptm, block, sptm)

    @pytest.mark.parametrize("polarizations", [[1.0], [1.0, -1.0], [0.3, -0.7, 0.0], [[0.5, 0.5], [-0.2, 1.0]]])
    def test_from_polarizations_entries(self, polarizations):
        polarizations = np.asarray(polarizations)
        covariance = CovarianceMatrixToolbox.from_polarizations(polarizations)
        n_modes = polarizations.shape[-1]
        expected = np.zeros(polarizations.shape[:-1] + (2 * n_modes, 2 * n_modes))
        for mode in range(n_modes):
            expected[..., 2 * mode, 2 * mode + 1] = -polarizations[..., mode]
            expected[..., 2 * mode + 1, 2 * mode] = polarizations[..., mode]
        np.testing.assert_allclose(covariance, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_from_polarizations_matches_basis_state_convention(self):
        bits = np.array([0, 1, 1, 0])
        expected = ProductState.from_basis_state(bits, wires=range(4)).covariance_matrix
        covariance = CovarianceMatrixToolbox.from_polarizations((-1.0) ** bits)
        np.testing.assert_allclose(
            covariance, np.asarray(expected), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_from_polarizations_preserves_backend_and_dtype(self, dtype):
        polarizations = torch.tensor([0.1, -0.4], dtype=dtype)
        covariance = CovarianceMatrixToolbox.from_polarizations(polarizations)
        assert isinstance(covariance, torch.Tensor)
        assert covariance.dtype == dtype
        assert isinstance(CovarianceMatrixToolbox.from_polarizations(np.array([0.1, -0.4])), np.ndarray)

    @pytest.mark.parametrize("n_modes, batch_size", [(1, None), (2, None), (4, None), (3, 2), (5, 3)])
    def test_block_diagonalize_reconstructs_input(self, n_modes, batch_size):
        covariance = self.random_covariance(n_modes, TEST_SEED, batch_size)
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("n_modes, batch_size", [(2, None), (4, None), (3, 2)])
    def test_block_diagonalize_returns_special_orthogonal_sptm(self, n_modes, batch_size):
        covariance = self.random_covariance(n_modes, TEST_SEED + 1, batch_size)
        sptm, _ = CovarianceMatrixToolbox.block_diagonalize(covariance)
        identity = np.broadcast_to(np.eye(2 * n_modes), sptm.shape)
        np.testing.assert_allclose(
            sptm @ np.swapaxes(sptm, -1, -2), identity, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            np.linalg.det(sptm), np.ones(sptm.shape[:-2]), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize("n_modes", [2, 3, 5])
    def test_block_diagonalize_produces_block_diagonal_form(self, n_modes):
        covariance = self.random_covariance(n_modes, TEST_SEED + 2)
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        rotated = sptm @ covariance @ sptm.T
        np.testing.assert_allclose(
            rotated,
            CovarianceMatrixToolbox.from_polarizations(polarizations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_block_diagonalize_recovers_polarization_magnitudes(self):
        rng = np.random.default_rng(TEST_SEED)
        expected = rng.uniform(-1.0, 1.0, size=4)
        covariance = self.random_covariance(4, TEST_SEED + 3)
        # The random covariance was built from the polarizations drawn first with the same seed.
        rotation = SingleParticleTransitionMatrixOperation.random_params(wires=list(range(4)), seed=TEST_SEED + 3).real
        covariance = rotation.T @ CovarianceMatrixToolbox.from_polarizations(expected) @ rotation
        _, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
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
            CovarianceMatrixToolbox.from_polarizations(np.array([1.0, 1.0, 1.0])),
            CovarianceMatrixToolbox.from_polarizations(np.array([1.0, -1.0, 0.0, 0.0])),
            CovarianceMatrixToolbox.from_polarizations(np.array([0.5, 0.5, -0.5])),
        ],
    )
    def test_block_diagonalize_handles_degenerate_inputs(self, covariance):
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            sptm @ sptm.T, np.eye(sptm.shape[-1]), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(np.linalg.det(sptm), 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_block_diagonalize_rotated_maximally_mixed_pair(self):
        rotation = SingleParticleTransitionMatrixOperation.random_params(wires=[0, 1, 2], seed=TEST_SEED).real
        covariance = rotation.T @ CovarianceMatrixToolbox.from_polarizations(np.array([0.0, 0.0, 1.0])) @ rotation
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            np.sort(np.abs(polarizations)), [0.0, 0.0, 1.0], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_block_diagonalize_preserves_backend_and_dtype(self, dtype):
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED), dtype=dtype)
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        assert isinstance(sptm, torch.Tensor) and sptm.dtype == dtype
        assert isinstance(polarizations, torch.Tensor) and polarizations.dtype == dtype
        sptm_np, polarizations_np = CovarianceMatrixToolbox.block_diagonalize(np.asarray(covariance))
        assert isinstance(sptm_np, np.ndarray) and isinstance(polarizations_np, np.ndarray)

    def test_block_diagonalize_accepts_complex_input_with_zero_imaginary_part(self):
        covariance = self.random_covariance(2, TEST_SEED).astype(complex)
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        np.testing.assert_allclose(
            self.reconstruct(sptm.real, polarizations.real),
            covariance.real,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_modes, batch_size", [(2, None), (3, None), (2, 2)])
    def test_block_diagonalize_gradient_of_reconstruction(self, n_modes, batch_size):
        covariance = torch.tensor(self.random_covariance(n_modes, TEST_SEED + 4, batch_size), requires_grad=True)

        def reconstruction(raw):
            antisymmetric = 0.5 * (raw - raw.mT)
            sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(antisymmetric)
            return self.reconstruct(sptm, polarizations)

        assert torch.autograd.gradcheck(
            reconstruction, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_block_diagonalize_gradient_of_gauge_invariant_loss(self):
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED + 5), requires_grad=True)
        weight = torch.tensor(np.random.default_rng(TEST_SEED).normal(size=(6, 6)))

        def loss(raw):
            antisymmetric = 0.5 * (raw - raw.mT)
            sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(antisymmetric)
            evolved = sptm.mT @ CovarianceMatrixToolbox.from_polarizations(polarizations) @ sptm
            return torch.sum(weight * evolved) + torch.sum(polarizations**2)

        assert torch.autograd.gradcheck(
            loss, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_block_diagonalize_gradient_at_pure_degenerate_point(self):
        # Every mode of a pure state has polarization +-1, a degenerate point of the decomposition. Along the
        # physical perturbations (rotations of the state) a loss on the reconstructed covariance matrix still
        # has a well-defined gradient, checked against finite differences with respect to the rotation angles.
        generator_indices = torch.triu_indices(6, 6, offset=1)
        initial = CovarianceMatrixToolbox.from_polarizations(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
        weight = torch.tensor(np.random.default_rng(TEST_SEED).normal(size=(6, 6)))

        def loss(angles):
            generator = torch.zeros(6, 6, dtype=angles.dtype)
            generator = generator.index_put((generator_indices[0], generator_indices[1]), angles)
            rotation = torch.linalg.matrix_exp(generator - generator.mT)
            covariance = rotation.mT @ initial @ rotation
            sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
            evolved = sptm.mT @ CovarianceMatrixToolbox.from_polarizations(polarizations) @ sptm
            return torch.sum(weight * evolved)

        angles = torch.tensor(np.random.default_rng(TEST_SEED + 1).normal(size=15), requires_grad=True)
        assert torch.autograd.gradcheck(loss, (angles,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True)

    def test_block_diagonalize_gradient_projects_gauge_directions(self):
        # The polarization gradient alone must map to a block-diagonal gradient in the rotated frame.
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED + 6), requires_grad=True)
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(covariance)
        (gradient,) = torch.autograd.grad(torch.sum(polarizations * torch.arange(1.0, 4.0)), covariance)
        rotated = (sptm @ gradient @ sptm.mT).detach().numpy()
        expected = np.zeros((6, 6))
        for mode in range(3):
            expected[2 * mode, 2 * mode + 1] = -0.5 * (mode + 1)
            expected[2 * mode + 1, 2 * mode] = 0.5 * (mode + 1)
        np.testing.assert_allclose(rotated, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_from_polarizations_promotes_integer_polarizations(self):
        covariance = CovarianceMatrixToolbox.from_polarizations([1, 0, -1])
        assert np.issubdtype(covariance.dtype, np.floating)
        np.testing.assert_allclose(
            covariance, CovarianceMatrixToolbox.from_polarizations([1.0, 0.0, -1.0]), atol=ATOL_MATRIX_COMPARISON
        )

    def test_block_diagonalize_promotes_integer_input(self):
        integer_covariance = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(integer_covariance)
        assert np.issubdtype(sptm.dtype, np.floating)
        assert np.issubdtype(polarizations.dtype, np.floating)
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations), integer_covariance, atol=ATOL_MATRIX_COMPARISON
        )

    def test_block_diagonalize_returns_real_for_complex_input(self):
        real_covariance = self.random_covariance(2, TEST_SEED + 7)
        sptm, polarizations = CovarianceMatrixToolbox.block_diagonalize(
            torch.tensor(real_covariance).to(torch.complex128)
        )
        assert sptm.dtype == torch.float64
        assert polarizations.dtype == torch.float64
        np.testing.assert_allclose(
            self.reconstruct(sptm, polarizations).numpy(),
            real_covariance,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @staticmethod
    def dense_entropy(density_matrix: np.ndarray) -> float:
        """Von Neumann entropy of a dense density matrix from its spectrum.

        :param density_matrix: Density matrix of shape ``(2^m, 2^m)``.
        :type density_matrix: np.ndarray
        :return: The entropy in nats.
        :rtype: float
        """
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    def test_displacement_vector_of_product_state(self):
        rng = np.random.default_rng(TEST_SEED)
        amplitudes = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        amplitudes = amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        state = ProductState(amplitudes, wires=range(3)).state_vector().reshape(-1)
        expected = [np.real(state.conj() @ get_majorana(index, 3) @ state) for index in range(6)]
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.displacement_vector(amplitudes, range(3)),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_displacement_vector_gradient(self):
        rng = np.random.default_rng(TEST_SEED)
        amplitudes = torch.tensor(rng.normal(size=(2, 3, 2)) + 1j * rng.normal(size=(2, 3, 2)), requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda raw: CovarianceMatrixToolbox.displacement_vector(raw, range(3)),
            (amplitudes,),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
        )

    def test_majorana_indices(self):
        np.testing.assert_array_equal(CovarianceMatrixToolbox.majorana_indices([2, 0]), [4, 5, 0, 1])

    @pytest.mark.parametrize(
        "wire_indices, expected",
        [([0], True), ([1, 2, 3], True), ([0, 2], False), ([1, 0], False), ([], False), ([2, 2], False)],
    )
    def test_are_wires_contiguous(self, wire_indices, expected):
        assert CovarianceMatrixToolbox.are_wires_contiguous(wire_indices) is expected

    @pytest.mark.parametrize("wire_indices", [[0], [1, 2], [0, 1, 2, 3]])
    def test_validate_reduced_wire_indices_accepts_contiguous_blocks(self, wire_indices):
        CovarianceMatrixToolbox.validate_reduced_wire_indices(wire_indices)

    @pytest.mark.parametrize("wire_indices", [[0, 2], [1, 0]])
    def test_validate_reduced_wire_indices_needs_opt_in_for_gaps(self, wire_indices):
        with pytest.raises(ValueError, match="not a contiguous block"):
            CovarianceMatrixToolbox.validate_reduced_wire_indices(wire_indices)
        CovarianceMatrixToolbox.validate_reduced_wire_indices(wire_indices, allow_gaps=True)

    @pytest.mark.parametrize("wire_indices, message", [([], "At least one wire"), ([1, 1], "distinct")])
    def test_validate_reduced_wire_indices_rejects_empty_and_repeated(self, wire_indices, message):
        with pytest.raises(ValueError, match=message):
            CovarianceMatrixToolbox.validate_reduced_wire_indices(wire_indices, allow_gaps=True)

    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_reduce_is_principal_submatrix(self, batch_size):
        covariance = self.random_covariance(4, TEST_SEED, batch_size)
        reduced = CovarianceMatrixToolbox.reduce(covariance, [1, 3])
        indices = [2, 3, 6, 7]
        np.testing.assert_array_equal(reduced, covariance[..., indices, :][..., :, indices])
        reduced_t = CovarianceMatrixToolbox.reduce(torch.tensor(covariance, dtype=torch.float32), [1, 3])
        assert isinstance(reduced_t, torch.Tensor) and reduced_t.dtype == torch.float32

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
    def test_mode_polarizations_recovers_magnitudes(self, batch_shape):
        rng = np.random.default_rng(TEST_SEED)
        expected = rng.uniform(-1.0, 1.0, size=batch_shape + (3,))
        rotation = SingleParticleTransitionMatrixOperation.random_params(wires=[0, 1, 2], seed=TEST_SEED).real
        covariance = np.einsum(
            "ij,...ik,kl->...jl", rotation, CovarianceMatrixToolbox.from_polarizations(expected), rotation
        )
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.mode_polarizations(covariance),
            np.sort(np.abs(expected), axis=-1),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_spectral_functions_preserve_backend_and_dtype(self, dtype):
        covariance = self.random_covariance(2, TEST_SEED)
        for function in (
            CovarianceMatrixToolbox.mode_polarizations,
            CovarianceMatrixToolbox.von_neumann_entropy,
            CovarianceMatrixToolbox.purity,
        ):
            result = function(torch.tensor(covariance, dtype=dtype))
            assert isinstance(result, torch.Tensor) and result.dtype == dtype
            assert isinstance(function(covariance), np.ndarray)

    @pytest.mark.parametrize("n_modes, batch_size", [(1, None), (2, None), (3, None), (3, 2)])
    def test_entropy_and_purity_match_dense_spectrum(self, n_modes, batch_size):
        covariance = self.random_covariance(n_modes, TEST_SEED + 8, batch_size)
        density = np.asarray(CovarianceMatrixToolbox.to_density_matrix(covariance))
        flat_density = density.reshape(-1, 2**n_modes, 2**n_modes)
        expected_entropy = np.array([self.dense_entropy(rho) for rho in flat_density])
        expected_purity = np.array([np.real(np.trace(rho @ rho)) for rho in flat_density])
        np.testing.assert_allclose(
            np.reshape(CovarianceMatrixToolbox.von_neumann_entropy(covariance), -1),
            expected_entropy,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            np.reshape(CovarianceMatrixToolbox.purity(covariance), -1),
            expected_purity,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_entropy_base(self):
        covariance = CovarianceMatrixToolbox.from_polarizations(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.von_neumann_entropy(covariance, base=2),
            2.0,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.von_neumann_entropy(covariance),
            2.0 * np.log(2.0),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_pure_state_has_zero_entropy_and_unit_purity(self):
        covariance = ProductState.from_basis_state(np.array([0, 1, 1]), wires=range(3)).covariance_matrix
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.von_neumann_entropy(covariance), 0.0, atol=ATOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(CovarianceMatrixToolbox.purity(covariance), 1.0, atol=ATOL_SCALAR_COMPARISON)

    def test_entropy_gradient_is_finite_for_a_pure_state(self):
        covariance = CovarianceMatrixToolbox.from_polarizations(
            torch.tensor([1.0, -1.0], dtype=torch.float64)
        ).requires_grad_(True)
        (gradient,) = torch.autograd.grad(CovarianceMatrixToolbox.von_neumann_entropy(covariance), covariance)
        assert torch.all(torch.isfinite(gradient))

    @pytest.mark.parametrize(
        "function",
        [
            CovarianceMatrixToolbox.von_neumann_entropy,
            CovarianceMatrixToolbox.purity,
            CovarianceMatrixToolbox.mode_polarizations,
        ],
    )
    def test_spectral_functions_gradient(self, function):
        covariance = torch.tensor(self.random_covariance(3, TEST_SEED + 9, 2), requires_grad=True)

        def antisymmetrized(raw):
            return function(0.5 * (raw - raw.mT))

        assert torch.autograd.gradcheck(
            antisymmetrized, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_to_density_matrix_of_product_of_mixtures(self):
        polarizations = np.array([0.4, -0.2])
        density = CovarianceMatrixToolbox.to_density_matrix(CovarianceMatrixToolbox.from_polarizations(polarizations))
        single_mode = [np.diag([(1 + value) / 2, (1 - value) / 2]) for value in polarizations]
        np.testing.assert_allclose(
            density, np.kron(*single_mode), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("bits", [[0, 0], [1, 0, 1]])
    def test_to_density_matrix_of_basis_state(self, bits):
        prep = ProductState.from_basis_state(np.array(bits), wires=range(len(bits)))
        state = np.asarray(prep.state_vector()).reshape(-1)
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.to_density_matrix(prep.covariance_matrix),
            np.outer(state, state.conj()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("n_modes, batch_size", [(1, None), (2, None), (3, None), (2, 3)])
    def test_density_matrix_round_trip(self, n_modes, batch_size):
        covariance = self.random_covariance(n_modes, TEST_SEED + 10, batch_size)
        density = CovarianceMatrixToolbox.to_density_matrix(covariance)
        np.testing.assert_allclose(
            np.trace(density, axis1=-2, axis2=-1), np.ones(covariance.shape[:-2]), atol=ATOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            density, np.conj(np.swapaxes(density, -1, -2)), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.from_density_matrix(density),
            covariance,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_from_density_matrix_matches_product_state(self):
        rng = np.random.default_rng(TEST_SEED)
        amplitudes = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        amplitudes = amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        prep = ProductState(amplitudes, wires=range(3))
        state = np.asarray(prep.state_vector()).reshape(-1)
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.from_density_matrix(np.outer(state, state.conj())),
            np.asarray(prep.covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "dtype, complex_dtype", [(torch.float32, torch.complex64), (torch.float64, torch.complex128)]
    )
    def test_density_matrix_conversions_preserve_backend_and_precision(self, dtype, complex_dtype):
        covariance = torch.tensor(self.random_covariance(2, TEST_SEED), dtype=dtype)
        density = CovarianceMatrixToolbox.to_density_matrix(covariance)
        assert isinstance(density, torch.Tensor) and density.dtype == complex_dtype
        recovered = CovarianceMatrixToolbox.from_density_matrix(density)
        assert isinstance(recovered, torch.Tensor) and recovered.dtype == dtype
        assert isinstance(CovarianceMatrixToolbox.to_density_matrix(covariance.numpy()), np.ndarray)
        assert isinstance(CovarianceMatrixToolbox.from_density_matrix(density.numpy()), np.ndarray)

    def test_density_matrix_conversions_gradient(self):
        covariance = torch.tensor(self.random_covariance(2, TEST_SEED + 11), requires_grad=True)

        def round_trip(raw):
            density = CovarianceMatrixToolbox.to_density_matrix(0.5 * (raw - raw.mT))
            return torch.real(density), torch.imag(density), CovarianceMatrixToolbox.from_density_matrix(density)

        assert torch.autograd.gradcheck(
            round_trip, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_from_polarizations_gradient(self):
        polarizations = torch.tensor([[0.3, -0.5], [0.1, 0.9]], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            CovarianceMatrixToolbox.from_polarizations,
            (polarizations,),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )

    def test_reduce_matches_qubit_partial_trace(self):
        covariance = self.random_covariance(3, TEST_SEED + 12)
        density = np.asarray(CovarianceMatrixToolbox.to_density_matrix(covariance))
        reduced_density = np.asarray(qml.math.partial_trace(density, [0]))
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.to_density_matrix(CovarianceMatrixToolbox.reduce(covariance, [1, 2])),
            reduced_density,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
