import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.wires import Wires

from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.operations.state_preparation import GaussianStatePrep, ProductState
from matchcake.utils.covariance_matrix_toolbox import CovarianceMatrixToolbox

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)


class TestGaussianStatePrep:
    @staticmethod
    def random_covariance(n_wires: int, batch_size=None, seed: int = TEST_SEED) -> np.ndarray:
        """Build a random mixed-state covariance matrix by rotating random polarizations.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :param seed: Seed of the random generator.
        :type seed: int
        :return: Real antisymmetric matrix of shape ``(2 * n_wires, 2 * n_wires)`` or batched.
        :rtype: np.ndarray
        """
        rng = np.random.default_rng(seed)
        shape = (n_wires,) if batch_size is None else (batch_size, n_wires)
        polarizations = rng.uniform(-0.9, 0.9, size=shape)
        rotation = SingleParticleTransitionMatrixOperation.random_params(
            batch_size=batch_size, wires=list(range(n_wires)), seed=seed
        ).real
        return np.einsum(
            "...ij,...ik,...kl->...jl",
            rotation,
            np.asarray(CovarianceMatrixToolbox.from_polarizations(polarizations)),
            rotation,
        )

    @staticmethod
    def dense_embedding(covariance: np.ndarray, wires, n_total: int) -> np.ndarray:
        """Density matrix of a Gaussian state on ``wires`` with the other wires in the zero state.

        :param covariance: Covariance matrix on ``wires``.
        :type covariance: np.ndarray
        :param wires: Wires of the Gaussian state.
        :param n_total: Number of wires of the register ``range(n_total)``.
        :type n_total: int
        :return: Density matrix of shape ``(2^n_total, 2^n_total)``.
        :rtype: np.ndarray
        """

        @qml.qnode(qml.device("default.mixed", wires=n_total))
        def circuit():
            qml.QubitDensityMatrix(CovarianceMatrixToolbox.to_density_matrix(covariance), wires=wires)
            return qml.density_matrix(wires=range(n_total))

        return np.asarray(circuit())

    @pytest.mark.parametrize("shape", [(4,), (3, 3), (6, 6), (4, 6), (2, 2, 4, 4)])
    def test_init_rejects_wrong_shapes(self, shape):
        with pytest.raises(ValueError, match="expects a covariance matrix of shape"):
            GaussianStatePrep(np.zeros(shape), wires=[0, 1])

    def test_init_rejects_non_antisymmetric_matrix(self):
        with pytest.raises(ValueError, match="antisymmetric"):
            GaussianStatePrep(np.eye(4), wires=[0, 1])

    def test_init_rejects_complex_matrix(self):
        covariance = self.random_covariance(2).astype(complex)
        with pytest.raises(ValueError, match="real"):
            GaussianStatePrep(covariance + 0.1j, wires=[0, 1])

    def test_init_accepts_complex_matrix_with_zero_imaginary_part(self):
        covariance = self.random_covariance(2)
        prep = GaussianStatePrep(covariance.astype(complex), wires=[0, 1])
        np.testing.assert_allclose(np.real(prep.covariance_matrix), covariance, atol=ATOL_MATRIX_COMPARISON)

    def test_init_rejects_unphysical_polarizations(self):
        with pytest.raises(ValueError, match="not the one of a physical state"):
            GaussianStatePrep.from_polarizations([3.0, 0.2], wires=[0, 1])

    def test_init_skips_validation_on_request(self):
        prep = GaussianStatePrep(np.eye(4), wires=[0, 1], validate=False)
        np.testing.assert_array_equal(prep.covariance_matrix, np.eye(4))

    def test_init_validates_torch_tensors_that_require_grad(self):
        covariance = torch.tensor(self.random_covariance(2), requires_grad=True)
        prep = GaussianStatePrep(covariance, wires=[0, 1])
        assert prep.covariance_matrix is covariance

    @pytest.mark.parametrize("batch_size", [None, 3])
    def test_covariance_matrix_and_batch_size(self, batch_size):
        covariance = self.random_covariance(2, batch_size)
        prep = GaussianStatePrep(covariance, wires=["a", "b"])
        np.testing.assert_array_equal(prep.covariance_matrix, covariance)
        assert prep.batch_size == batch_size
        assert prep.wires == Wires(["a", "b"])
        assert prep.num_params == 1
        assert prep.ndim_params == (2,)

    def test_from_polarizations_without_sptm(self):
        polarizations = np.array([0.3, -1.0, 0.0])
        prep = GaussianStatePrep.from_polarizations(polarizations, wires=range(3))
        np.testing.assert_allclose(
            prep.covariance_matrix,
            CovarianceMatrixToolbox.from_polarizations(polarizations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_from_polarizations_with_sptm(self, batch_size):
        rng = np.random.default_rng(TEST_SEED)
        polarizations = rng.uniform(-1.0, 1.0, size=(2,) if batch_size is None else (batch_size, 2))
        sptm = SingleParticleTransitionMatrixOperation.random_params(
            batch_size=batch_size, wires=[0, 1], seed=TEST_SEED
        ).real
        prep = GaussianStatePrep.from_polarizations(polarizations, wires=[0, 1], sptm=sptm)
        expected = np.einsum(
            "...ji,...jk,...kl->...il", sptm, CovarianceMatrixToolbox.from_polarizations(polarizations), sptm
        )
        np.testing.assert_allclose(
            prep.covariance_matrix, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_from_polarizations_of_basis_state_matches_product_state(self):
        bits = np.array([1, 0, 1])
        prep = GaussianStatePrep.from_polarizations((-1.0) ** bits, wires=range(3))
        expected = ProductState.from_basis_state(bits, wires=range(3)).covariance_matrix
        np.testing.assert_allclose(prep.covariance_matrix, np.asarray(expected), atol=ATOL_MATRIX_COMPARISON)

    def test_decomposition_is_the_dense_density_matrix(self):
        covariance = self.random_covariance(2)
        (operation,) = GaussianStatePrep(covariance, wires=[0, 1]).decomposition()
        assert isinstance(operation, qml.QubitDensityMatrix)
        assert operation.wires == Wires([0, 1])
        np.testing.assert_allclose(
            operation.parameters[0],
            CovarianceMatrixToolbox.to_density_matrix(covariance),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_decomposition_rejects_batched_state(self):
        prep = GaussianStatePrep(self.random_covariance(2, batch_size=2), wires=[0, 1])
        with pytest.raises(ValueError, match="batched"):
            prep.decomposition()

    def test_state_vector_is_not_defined(self):
        prep = GaussianStatePrep(self.random_covariance(2), wires=[0, 1])
        with pytest.raises(NotImplementedError, match="no state vector"):
            prep.state_vector()

    def test_to_wires_returns_self_for_same_wires(self):
        prep = GaussianStatePrep(self.random_covariance(2), wires=[0, 1])
        assert prep.to_wires([0, 1]) is prep

    def test_to_wires_rejects_missing_wires(self):
        prep = GaussianStatePrep(self.random_covariance(2), wires=[0, 4])
        with pytest.raises(ValueError, match="do not contain the wires"):
            prep.to_wires([0, 1, 2])

    def test_to_wires_rejects_reordered_wires(self):
        prep = GaussianStatePrep(self.random_covariance(2), wires=[1, 0])
        with pytest.raises(ValueError, match="same order"):
            prep.to_wires([0, 1, 2])

    @pytest.mark.parametrize("wires, n_total", [([0, 1], 3), ([1, 2], 3), ([0, 2], 3), ([1, 3], 4), ([2], 3)])
    def test_to_wires_pads_with_the_zero_state(self, wires, n_total):
        covariance = self.random_covariance(len(wires))
        extended = GaussianStatePrep(covariance, wires=wires).to_wires(range(n_total))
        assert extended.wires == Wires(range(n_total))
        np.testing.assert_allclose(
            CovarianceMatrixToolbox.to_density_matrix(extended.covariance_matrix),
            self.dense_embedding(covariance, wires, n_total),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_to_wires_keeps_batch_backend_and_dtype(self):
        covariance = torch.tensor(self.random_covariance(2, batch_size=3), dtype=torch.float32)
        extended = GaussianStatePrep(covariance, wires=[1, 2]).to_wires(range(4)).covariance_matrix
        assert isinstance(extended, torch.Tensor) and extended.dtype == torch.float32
        assert tuple(extended.shape) == (3, 8, 8)
        np.testing.assert_allclose(extended[:, 2:6, 2:6].numpy(), covariance.numpy(), atol=ATOL_MATRIX_COMPARISON)

    def test_to_wires_gradient(self):
        covariance = torch.tensor(self.random_covariance(2, batch_size=2), requires_grad=True)

        def extend(raw):
            antisymmetric = 0.5 * (raw - raw.mT)
            return GaussianStatePrep(antisymmetric, wires=[0, 2]).to_wires(range(3)).covariance_matrix

        assert torch.autograd.gradcheck(
            extend, (covariance,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )
