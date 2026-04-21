from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import pennylane as qml

from matchcake import MatchgateOperation, NonInteractingFermionicDevice
from matchcake.ml.kernels import NIFKernel
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.utils.math import dagger, matmul


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

    def test_forward(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance(x)
        kernel_instance.ansatz_mock.assert_called()

    def test_compute_similarities(self, kernel_instance):
        x = torch.rand(10, 10)
        kernel_instance.compute_similarities(x, x)
        kernel_instance.ansatz_mock.assert_called()

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
        sptm1 = SingleParticleTransitionMatrixOperation.random_params(batch_size=10, seed=seed+1, wires=wires)
        kernel = NIFKernel(n_qubits=len(wires))

        def qnode_wrapper(*args, **kwargs):
            _ = list(kernel.circuit(*args, **kwargs))
            return qml.probs()

        qnode = qml.QNode(qnode_wrapper, kernel.q_device)
        _ = qnode(sptm0, sptm1)
        tape = qnode._tape.circuit[:-1]
        assert len(tape) == 2, f"Expected 2 operations, got {len(tape)} from {tape}."

        global_sptm = kernel.q_device.global_sptm.matrix()
        np.testing.assert_allclose(matmul(sptm0, dagger(sptm1)), global_sptm)


