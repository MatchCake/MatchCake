import numpy as np
import pennylane as qml
import pytest
import torch
from pennylane.wires import Wires

import matchcake as mc
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

    @staticmethod
    def product_state(n_wires: int, batch_size=None, seed: int = TEST_SEED) -> ProductState:
        """Build a random product state with superposed qubits, which has a displacement vector.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :param seed: Seed of the random generator.
        :type seed: int
        :return: The product state on ``range(n_wires)``.
        :rtype: ProductState
        """
        rng = np.random.default_rng(seed)
        shape = (n_wires, 2) if batch_size is None else (batch_size, n_wires, 2)
        amplitudes = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        amplitudes = amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        return ProductState(torch.as_tensor(amplitudes), wires=range(n_wires))

    def test_no_displacement_by_default(self):
        op = GaussianStatePrep(self.random_covariance(2), wires=[0, 1])
        assert op.num_params == 1
        assert op.ndim_params == (2,)
        assert len(op.data) == 1
        np.testing.assert_array_equal(qml.math.toarray(op.displacement_vector), np.zeros(4))

    @pytest.mark.parametrize("batch_size", [None, 3])
    def test_displacement_vector_and_batch_size(self, batch_size):
        product_state = self.product_state(3, batch_size)
        op = GaussianStatePrep(
            product_state.covariance_matrix, wires=[0, 1, 2], displacement_vector=product_state.displacement_vector
        )
        assert op.num_params == 2
        assert op.ndim_params == (2, 1)
        assert op.batch_size == batch_size
        np.testing.assert_allclose(
            qml.math.toarray(op.displacement_vector),
            qml.math.toarray(product_state.displacement_vector),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            qml.math.toarray(op.covariance_matrix),
            qml.math.toarray(product_state.covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("shape", [(3,), (6,), (2, 2), (1, 2, 4)])
    def test_init_rejects_wrong_displacement_shapes(self, shape):
        with pytest.raises(ValueError, match="displacement vector of shape"):
            GaussianStatePrep(np.zeros((4, 4)), wires=[0, 1], displacement_vector=np.zeros(shape))

    def test_init_rejects_mismatched_batch_sizes(self):
        with pytest.raises(ValueError, match="same batch size"):
            GaussianStatePrep(np.zeros((2, 4, 4)), wires=[0, 1], displacement_vector=np.zeros((3, 4)))

    def test_init_repeats_the_unbatched_parameter(self):
        covariance = self.random_covariance(2)
        displacements = 0.01 * np.arange(12.0).reshape(3, 4)
        op = GaussianStatePrep(covariance, wires=[0, 1], displacement_vector=displacements)
        assert tuple(qml.math.shape(op.covariance_matrix)) == (3, 4, 4)
        np.testing.assert_array_equal(qml.math.toarray(op.covariance_matrix)[1], covariance)
        op = GaussianStatePrep(np.stack([covariance] * 3), wires=[0, 1], displacement_vector=displacements[0])
        assert tuple(qml.math.shape(op.displacement_vector)) == (3, 4)
        np.testing.assert_array_equal(qml.math.toarray(op.displacement_vector)[2], displacements[0])
        assert op.batch_size == 3

    def test_init_rejects_unphysical_displacement(self):
        covariance = np.asarray(CovarianceMatrixToolbox.from_polarizations(np.ones(2)))
        displacement = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="bordered matrix"):
            GaussianStatePrep(covariance, wires=[0, 1], displacement_vector=displacement)
        op = GaussianStatePrep(covariance, wires=[0, 1], validate=False, displacement_vector=displacement)
        assert op.num_params == 2

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_init_rejects_a_displacement_that_is_not_finite(self, value):
        with pytest.raises(ValueError, match="must be finite"):
            GaussianStatePrep(self.random_covariance(2), wires=[0, 1], displacement_vector=np.full(4, value))

    @pytest.mark.parametrize("magnitude, is_zero", [(1e-13, True), (1e-8, False), (1e-3, False)])
    def test_decomposition_uses_the_displacement_tolerance_of_the_device(self, magnitude, is_zero):
        displacement = np.array([magnitude, 0.0, 0.0, 0.0])
        op = GaussianStatePrep(np.zeros((4, 4)), wires=[0, 1], displacement_vector=displacement)
        device = mc.NonInteractingFermionicDevice(wires=2)
        device.apply([op])
        assert device._has_displacement() is not is_zero
        if is_zero:
            assert len(op.decomposition()) == 1
        else:
            with pytest.raises(NotImplementedError, match="nonzero displacement vector"):
                op.decomposition()

    def test_equal_sees_the_displacement_vector(self):
        product_state = self.product_state(2)
        covariance, displacement = product_state.covariance_matrix, product_state.displacement_vector
        displaced = GaussianStatePrep(covariance, wires=[0, 1], displacement_vector=displacement)
        plain = GaussianStatePrep(covariance, wires=[0, 1])
        assert not qml.equal(displaced, plain)
        assert not qml.equal(plain, displaced)
        assert qml.equal(displaced, GaussianStatePrep(covariance, wires=[0, 1], displacement_vector=displacement))
        assert not qml.equal(
            displaced, GaussianStatePrep(covariance, wires=[0, 1], validate=False, displacement_vector=-displacement)
        )
        assert qml.equal(plain, GaussianStatePrep(covariance, wires=[0, 1]))
        assert not qml.equal(plain, GaussianStatePrep(covariance, wires=[1, 0]))

    def test_init_accepts_the_displacement_of_a_pure_product_state(self):
        product_state = self.product_state(3)
        GaussianStatePrep(
            product_state.covariance_matrix, wires=[0, 1, 2], displacement_vector=product_state.displacement_vector
        )

    def test_init_rejects_complex_displacement(self):
        with pytest.raises(ValueError, match="displacement vector must be real"):
            GaussianStatePrep(np.zeros((4, 4)), wires=[0, 1], displacement_vector=0.1j * np.ones(4))
        op = GaussianStatePrep(np.zeros((4, 4)), wires=[0, 1], displacement_vector=0.1 * np.ones(4) + 0.0j)
        assert not np.iscomplexobj(qml.math.toarray(op.displacement_vector))

    def test_init_converts_both_parameters_to_torch(self):
        covariance = self.random_covariance(2).astype(np.float32)
        displacement = torch.full((4,), 0.01, dtype=torch.float64, requires_grad=True)
        op = GaussianStatePrep(covariance, wires=[0, 1], displacement_vector=displacement)
        assert isinstance(op.covariance_matrix, torch.Tensor)
        assert op.displacement_vector.dtype == torch.float32
        assert op.displacement_vector.requires_grad
        op = GaussianStatePrep(torch.as_tensor(covariance), wires=[0, 1], displacement_vector=np.full(4, 0.01))
        assert isinstance(op.displacement_vector, torch.Tensor)
        assert op.displacement_vector.dtype == torch.float32

    def test_decomposition_rejects_a_displaced_state(self):
        product_state = self.product_state(2)
        op = GaussianStatePrep(
            product_state.covariance_matrix, wires=[0, 1], displacement_vector=product_state.displacement_vector
        )
        with pytest.raises(NotImplementedError, match="nonzero displacement vector"):
            op.decomposition()

    def test_decomposition_with_a_zero_displacement(self):
        covariance = self.random_covariance(2)
        (with_displacement,) = GaussianStatePrep(
            covariance, wires=[0, 1], displacement_vector=np.zeros(4)
        ).decomposition()
        (without_displacement,) = GaussianStatePrep(covariance, wires=[0, 1]).decomposition()
        np.testing.assert_allclose(
            with_displacement.parameters[0],
            without_displacement.parameters[0],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("batch_size", [None, 2])
    @pytest.mark.parametrize("wires, n_total", [([0, 1], 4), ([1, 2], 4), ([2, 3], 4), ([0, 2], 4), ([1, 3], 5)])
    def test_to_wires_pads_the_displacement_with_the_zero_state(self, wires, n_total, batch_size):
        product_state = self.product_state(len(wires), batch_size)
        op = GaussianStatePrep(
            product_state.covariance_matrix, wires=wires, displacement_vector=product_state.displacement_vector
        )
        padded = op.to_wires(range(n_total))
        amplitudes = product_state.data[0]
        zero_state = torch.zeros(tuple(amplitudes.shape[:-2]) + (n_total, 2), dtype=amplitudes.dtype)
        zero_state[..., 0] = 1.0
        zero_state[..., wires, :] = amplitudes
        expected = ProductState(zero_state, wires=range(n_total))
        assert padded.num_params == 2
        assert padded.wires == Wires(range(n_total))
        np.testing.assert_allclose(
            qml.math.toarray(padded.displacement_vector),
            qml.math.toarray(expected.displacement_vector),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            qml.math.toarray(padded.covariance_matrix),
            qml.math.toarray(expected.covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_to_wires_without_displacement_stays_without_displacement(self):
        padded = GaussianStatePrep(self.random_covariance(2), wires=[1, 2]).to_wires(range(4))
        assert padded.num_params == 1

    def test_to_wires_displacement_gradient(self):
        product_state = self.product_state(2)
        covariance = product_state.covariance_matrix.detach()
        displacement = product_state.displacement_vector.detach().clone().requires_grad_(True)

        def padded_displacement(vector):
            op = GaussianStatePrep(covariance, wires=[1, 2], validate=False, displacement_vector=vector)
            return op.to_wires(range(4)).displacement_vector

        assert torch.autograd.gradcheck(padded_displacement, (displacement,), eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL)

    def test_parameters_can_be_rebound(self):
        product_state = self.product_state(2)
        op = GaussianStatePrep(
            product_state.covariance_matrix, wires=[3, 5], displacement_vector=product_state.displacement_vector
        )
        new_parameters = [0.5 * parameter for parameter in op.data]
        rebound = qml.ops.functions.bind_new_parameters(op, new_parameters)
        assert isinstance(rebound, GaussianStatePrep)
        assert rebound.wires == op.wires
        assert rebound.num_params == 2
        np.testing.assert_allclose(
            qml.math.toarray(rebound.displacement_vector),
            0.5 * qml.math.toarray(op.displacement_vector),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        plain = GaussianStatePrep(self.random_covariance(2), wires=[3, 5])
        rebound_plain = qml.ops.functions.bind_new_parameters(plain, [0.5 * plain.data[0]])
        assert rebound_plain.num_params == 1
        np.testing.assert_allclose(
            qml.math.toarray(rebound_plain.covariance_matrix),
            0.5 * qml.math.toarray(plain.covariance_matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("with_displacement", [True, False])
    def test_flatten_and_unflatten(self, with_displacement):
        product_state = self.product_state(2)
        displacement = product_state.displacement_vector if with_displacement else None
        op = GaussianStatePrep(
            product_state.covariance_matrix, wires=[0, 1], validate=False, displacement_vector=displacement
        )
        data, metadata = op._flatten()
        rebuilt = GaussianStatePrep._unflatten(data, metadata)
        assert rebuilt.num_params == op.num_params
        assert rebuilt.wires == op.wires
        for rebuilt_parameter, parameter in zip(rebuilt.data, op.data, strict=True):
            np.testing.assert_array_equal(qml.math.toarray(rebuilt_parameter), qml.math.toarray(parameter))
