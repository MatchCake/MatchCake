import numpy as np
import pytest
import torch
from pennylane.wires import Wires

from matchcake.operations.state_preparation import (
    GaussianStatePrep,
    GaussianStatePrepBase,
    ProductState,
)
from matchcake.utils.majorana import get_majorana

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)


class TestGaussianStatePrepBase:
    @staticmethod
    def random_amplitudes(n_wires: int, batch_size=None) -> np.ndarray:
        """Draw normalized per-qubit amplitudes.

        :param n_wires: Number of qubits.
        :type n_wires: int
        :param batch_size: Optional leading batch size.
        :type batch_size: Optional[int]
        :return: Complex amplitudes of shape ``(n_wires, 2)`` or ``(batch_size, n_wires, 2)``.
        :rtype: np.ndarray
        """
        rng = np.random.default_rng(TEST_SEED)
        shape = (() if batch_size is None else (batch_size,)) + (n_wires, 2)
        amplitudes = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)

    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            GaussianStatePrepBase(np.zeros((4, 4)), wires=[0, 1])

    @pytest.mark.parametrize("state_prep_class", [ProductState, GaussianStatePrep])
    def test_state_preparations_share_the_base(self, state_prep_class):
        assert issubclass(state_prep_class, GaussianStatePrepBase)

    @pytest.mark.parametrize("batch_shape", [(), (3,)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_default_displacement_vector_is_zero(self, batch_shape, dtype):
        covariance = torch.zeros(batch_shape + (4, 4), dtype=dtype)
        displacement = GaussianStatePrep(covariance, wires=[0, 1]).displacement_vector
        assert isinstance(displacement, torch.Tensor) and displacement.dtype == dtype
        assert tuple(displacement.shape) == batch_shape + (4,)
        np.testing.assert_array_equal(displacement.numpy(), 0.0)

    def test_default_displacement_vector_of_numpy_covariance(self):
        displacement = GaussianStatePrep(np.zeros((4, 4)), wires=[0, 1]).displacement_vector
        np.testing.assert_array_equal(np.asarray(displacement), np.zeros(4))

    def test_product_state_displacement_vector_matches_dense(self):
        n_wires = 3
        prep = ProductState(self.random_amplitudes(n_wires), wires=range(n_wires))
        state = np.asarray(prep.state_vector()).reshape(-1)
        expected = [np.real(state.conj() @ get_majorana(index, n_wires) @ state) for index in range(2 * n_wires)]
        np.testing.assert_allclose(
            prep.displacement_vector, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_product_state_displacement_vector_is_batched(self):
        prep = ProductState(self.random_amplitudes(3, batch_size=2), wires=range(3))
        assert tuple(prep.displacement_vector.shape) == (2, 6)

    def test_basis_state_has_no_displacement(self):
        prep = ProductState.from_basis_state(np.array([1, 0, 1]), wires=range(3))
        np.testing.assert_allclose(prep.displacement_vector, np.zeros(6), atol=ATOL_MATRIX_COMPARISON)

    def test_to_wires_is_part_of_the_interface(self):
        prep = ProductState.from_basis_state(np.array([1]), wires=[1])
        extended = prep.to_wires([0, 1])
        assert isinstance(extended, GaussianStatePrepBase)
        assert extended.wires == Wires([0, 1])
