from unittest.mock import MagicMock

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake.ml.kernels import NIFKernel
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.utils.math import dagger, matmul

from ...configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON


class TestNIFKernel:
    @pytest.fixture
    def kernel_instance(self):
        instance = NIFKernel()
        instance.ansatz_mock = MagicMock()

        def ansatz_mock_func(x):
            instance.ansatz_mock(x)
            yield MatchgateOperation.random(wires=[0, 1], batch_size=len(x))
            return

        instance.ansatz = ansatz_mock_func
        return instance

    def test_init(self, kernel_instance):
        assert kernel_instance.random_state == 0
        assert kernel_instance.gram_batch_size == 10_000
        assert isinstance(kernel_instance.q_device, NonInteractingFermionicDevice)

    def test_default_dtypes_match_device(self):
        kernel = NIFKernel()
        assert kernel.R_DTYPE == NIFKernel.DEFAULT_R_DTYPE
        assert kernel.C_DTYPE == NIFKernel.DEFAULT_C_DTYPE
        assert kernel.q_device.R_DTYPE == kernel.R_DTYPE
        assert kernel.q_device.C_DTYPE == kernel.C_DTYPE

    @pytest.mark.parametrize(
        "r_dtype, c_dtype",
        [
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
        ],
    )
    def test_custom_dtypes_propagate_to_device(self, r_dtype, c_dtype):
        kernel = NIFKernel(r_dtype=r_dtype, c_dtype=c_dtype)
        assert kernel.R_DTYPE == r_dtype
        assert kernel.C_DTYPE == c_dtype
        assert kernel.q_device.R_DTYPE == r_dtype
        assert kernel.q_device.C_DTYPE == c_dtype

    def test_dtypes_preserved_when_setting_n_qubits(self):
        kernel = NIFKernel(r_dtype=torch.float64, c_dtype=torch.complex128)
        kernel.n_qubits = 4
        assert kernel.q_device.R_DTYPE == torch.float64
        assert kernel.q_device.C_DTYPE == torch.complex128

    def test_sklearn_clone_preserves_dtypes(self):
        from sklearn.base import clone

        kernel = NIFKernel(r_dtype=torch.float64, c_dtype=torch.complex128)
        assert kernel.get_params()["r_dtype"] == torch.float64
        assert kernel.get_params()["c_dtype"] == torch.complex128
        cloned = clone(kernel)
        assert cloned.R_DTYPE == torch.float64
        assert cloned.C_DTYPE == torch.complex128

    def test_forward(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance(x)
        kernel_instance.ansatz_mock.assert_called()

    def test_compute_similarities(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance.compute_similarities(x, x)
        kernel_instance.ansatz_mock.assert_called()

    @pytest.mark.parametrize("shape_0, shape_1", [(0, 4), (4, 0), (0, 0)])
    def test_forward_with_empty_inputs(self, kernel_instance, shape_0, shape_1):
        kernel = kernel_instance(torch.rand(shape_0, 10), torch.rand(shape_1, 10))
        assert kernel.shape == (shape_0, shape_1)
        assert kernel.dtype == kernel_instance.R_DTYPE
        kernel_instance.ansatz_mock.assert_not_called()

    def test_forward_with_one_empty_input(self, kernel_instance):
        kernel = kernel_instance(torch.rand(0, 10))
        assert kernel.shape == (0, 0)

    def test_transform_with_empty_input(self, kernel_instance):
        kernel_instance.fit(torch.rand(4, 10))
        assert kernel_instance.transform(torch.rand(0, 10)).shape == (0, 4)

    def test_fit_drops_cached_train_sptms(self, kernel_instance):
        x_test = torch.rand(3, 10)
        kernel_instance.fit(torch.rand(4, 10))
        assert kernel_instance.transform(x_test).shape == (3, 4)
        first_sptms_train = kernel_instance.sptms_train
        kernel_instance.fit(torch.rand(6, 10))
        assert kernel_instance._sptms_train_cache is None
        assert kernel_instance.transform(x_test).shape == (3, 6)
        assert kernel_instance.sptms_train.shape[0] == 6
        assert kernel_instance.sptms_train is not first_sptms_train

    def test_forward_with_empty_input_keeps_dtype(self):
        kernel_instance = NIFKernel(r_dtype=torch.float64, c_dtype=torch.complex128)
        kernel = kernel_instance(torch.rand(0, 10), torch.rand(4, 10))
        assert kernel.shape == (0, 4)
        assert kernel.dtype == torch.float64

    def test_fit_on_zero_samples_raises(self, kernel_instance):
        with pytest.raises(ValueError, match="zero samples"):
            kernel_instance.fit(torch.rand(0, 10))
        assert not kernel_instance.is_fitted_

    def test_failed_fit_drops_cached_train_sptms(self, kernel_instance):
        kernel_instance.fit(torch.rand(4, 10))
        kernel_instance.transform(torch.rand(3, 10))
        assert kernel_instance._sptms_train_cache is not None
        kernel_instance.alignment = True
        with pytest.raises(ValueError):
            kernel_instance.fit(torch.rand(6, 10))
        assert kernel_instance._sptms_train_cache is None

    def test_are_same_samples_with_numpy_inputs(self):
        x = np.random.rand(5, 4)
        assert NIFKernel.are_same_samples(x, x.copy())
        assert NIFKernel.are_same_samples(torch.from_numpy(x), x.copy())
        assert not NIFKernel.are_same_samples(x, x + 1.0)

    def test_sptms_train_is_cached_between_transforms(self, kernel_instance):
        kernel_instance.fit(torch.rand(4, 10))
        kernel_instance.transform(torch.rand(3, 10))
        sptms_train = kernel_instance.sptms_train
        kernel_instance.ansatz_mock.reset_mock()
        kernel_instance.transform(torch.rand(2, 10))
        assert kernel_instance.sptms_train is sptms_train
        assert kernel_instance.ansatz_mock.call_count == 1

    def test_sptms_train_follows_x_train(self, kernel_instance):
        kernel_instance.fit(torch.rand(4, 10))
        assert kernel_instance.transform(torch.rand(3, 10)).shape == (3, 4)
        kernel_instance.x_train_ = torch.rand(6, 10)
        assert kernel_instance.transform(torch.rand(3, 10)).shape == (3, 6)

    def test_sptms_train_without_cache_attribute(self, kernel_instance):
        kernel_instance.fit(torch.rand(4, 10))
        del kernel_instance._sptms_train_cache
        assert kernel_instance.transform(torch.rand(3, 10)).shape == (3, 4)

    def test_sptms_train_follows_n_qubits(self, kernel_instance):
        kernel_instance.fit(torch.rand(4, 10))
        kernel_instance.transform(torch.rand(3, 10))
        kernel_instance.n_qubits = 3
        assert kernel_instance.transform(torch.rand(3, 10)).shape == (3, 4)
        assert kernel_instance.sptms_train.shape == (4, 6, 6)

    def test_are_same_samples(self):
        x = torch.rand(5, 4)
        assert NIFKernel.are_same_samples(x, x)
        assert NIFKernel.are_same_samples(x, x.clone())
        assert NIFKernel.are_same_samples(x, x.clone().to(torch.float64))
        assert not NIFKernel.are_same_samples(x, x[:3])
        assert not NIFKernel.are_same_samples(x, torch.flip(x, dims=(0,)))
        assert not NIFKernel.are_same_samples(x, x + 1.0)
        integers = torch.randint(0, 4, (5, 4))
        assert NIFKernel.are_same_samples(integers, integers.to(torch.float32))
        assert not NIFKernel.are_same_samples(integers, integers.to(torch.float32) + 0.5)
        assert not NIFKernel.are_same_samples(x, torch.nextafter(x, torch.ones_like(x)))

    @pytest.mark.parametrize("shape_0, shape_1", [(5, 5), (6, 3), (3, 6), (1, 1)])
    def test_compute_similarities_evaluates_every_pair_of_a_cross_kernel(self, kernel_instance, shape_0, shape_1):
        x0, x1 = torch.rand(shape_0, 10), torch.rand(shape_1, 10)
        evaluated = []

        def similarities_func(indices, sptm0, sptm1, p_bar):
            evaluated.append(indices)
            return 10.0 * indices[:, 0] + indices[:, 1] + 0.5

        kernel_instance._compute_similarities_func = similarities_func
        kernel = kernel_instance.compute_similarities(x0, x1)
        expected = torch.tensor([[10.0 * i + j + 0.5 for j in range(shape_1)] for i in range(shape_0)])
        torch.testing.assert_close(kernel, expected.to(kernel.dtype))
        assert len(np.concatenate(evaluated, axis=0)) == shape_0 * shape_1

    def test_compute_similarities_evaluates_one_triangle_for_same_samples(self, kernel_instance):
        x = torch.rand(5, 10)
        evaluated = []

        def similarities_func(indices, sptm0, sptm1, p_bar):
            evaluated.append(indices)
            return 0.1 * (indices[:, 0] + indices[:, 1])

        kernel_instance._compute_similarities_func = similarities_func
        kernel = kernel_instance.compute_similarities(x, x.clone())
        assert len(np.concatenate(evaluated, axis=0)) == 10
        torch.testing.assert_close(kernel, kernel.T)
        torch.testing.assert_close(torch.diagonal(kernel), torch.ones(5, dtype=kernel.dtype))

    @pytest.mark.parametrize("n_qubits, seed", list(zip(np.arange(2, 12), np.arange(10))))
    def test_queuing(self, n_qubits, seed):
        """
        Tests the queuing functionality of the `SingleParticleTransitionMatrixOperation`
        and its integration with the `NIFKernel` quantum kernel. Assures that the tape of
        the circuit contains the expected number of operations and that the global single-particle
        transition matrix matches the expected computed value.

        The test uses parameterized input to verify the behavior over a range of qubit numbers and
        seeds for pseudo-random number generation.

        :param n_qubits: Number of qubits in the quantum device, representing a range.
        :type n_qubits: int
        :param seed: Seed for pseudo-random number generation to ensure reproducibility.
        :type seed: int
        :return: None
        """
        wires = list(range(n_qubits))
        sptm0 = SingleParticleTransitionMatrixOperation.random_params(batch_size=10, seed=seed, wires=wires)
        sptm1 = SingleParticleTransitionMatrixOperation.random_params(batch_size=10, seed=seed + 1, wires=wires)
        kernel = NIFKernel(n_qubits=len(wires))

        def qnode_wrapper(*args, **kwargs):
            _ = list(kernel.circuit(*args, **kwargs))
            return qml.probs()

        qnode = qml.QNode(qnode_wrapper, kernel.q_device)
        _ = qnode(sptm0, sptm1)
        tape = qnode._tape.circuit[:-1]
        assert len(tape) == 2, f"Expected 2 operations, got {len(tape)} from {tape}."

        global_sptm = kernel.q_device.global_sptm.sptm_matrix()
        np.testing.assert_allclose(
            matmul(sptm0, dagger(sptm1)), global_sptm, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
