import numpy as np
import pennylane as qml
import pytest
import torch.nn

from matchcake.ml.kernels.fermionic_pqc_kernel import (
    PATTERN_TO_WIRES,
    FermionicPQCKernel,
)
from matchcake.ml.kernels.gram_matrix import GramMatrix
from matchcake.operations import CompHH, MAngleEmbedding, fSWAP


@pytest.mark.parametrize(
    "n_qubits, rotations, entangling_mth",
    [
        (n_qubits, rot, ent)
        for n_qubits in [4, 8]
        for rot in [
            "Y,Z",
            "X,Y,Z",
            "X",
            "Y",
            "Z",
        ]
        for ent in ["identity", "fswap", "hadamard"]
    ],
)
class TestFermionicPQCKernel:
    @pytest.fixture(scope="function")
    def kernel_instance(self, n_qubits, rotations, entangling_mth):
        instance = FermionicPQCKernel(n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth)
        return instance

    def test_custom_dtypes_propagate_to_device(self, n_qubits, rotations, entangling_mth):
        kernel = FermionicPQCKernel(
            n_qubits=n_qubits,
            rotations=rotations,
            entangling_mth=entangling_mth,
            r_dtype=torch.float64,
            c_dtype=torch.complex128,
        )
        assert kernel.R_DTYPE == torch.float64
        assert kernel.C_DTYPE == torch.complex128
        assert kernel.q_device.R_DTYPE == torch.float64
        assert kernel.q_device.C_DTYPE == torch.complex128

    @pytest.fixture
    def x_train(self):
        return np.linspace(0.0, 1.0, num=80).reshape(10, 8)

    @pytest.fixture
    def state_vector_kernel(self, n_qubits, rotations, entangling_mth):
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits,
            rotations=rotations,
            entangling_mth=entangling_mth,
            gram_batch_size=1,
        )
        kernel_instance._q_device = qml.device("default.qubit", wires=kernel_instance.n_qubits)

        def ansatz(x):
            wires_double = PATTERN_TO_WIRES["double"](kernel_instance.wires)
            wires_double_odd = PATTERN_TO_WIRES["double_odd"](kernel_instance.wires)
            wires_patterns = [wires_double, wires_double_odd]
            for layer in range(kernel_instance.depth_):
                sub_x = x[..., layer * kernel_instance.n_qubits : (layer + 1) * kernel_instance.n_qubits]
                MAngleEmbedding(sub_x, wires=kernel_instance.wires, rotations=kernel_instance.rotations)
                wires_list = wires_patterns[layer % len(wires_patterns)]
                for wires in wires_list:
                    if kernel_instance.entangling_mth == "fswap":
                        fSWAP(wires=wires)
                    elif kernel_instance.entangling_mth == "hadamard":
                        CompHH(wires=wires)
                    elif kernel_instance.entangling_mth == "identity":
                        pass
                    else:
                        raise ValueError(f"Unknown entangling method: {kernel_instance.entangling_mth}")
            return

        def circuit(x0, x1):
            ansatz(x0)
            qml.adjoint(ansatz)(x1)
            projector = qml.Projector(np.zeros(kernel_instance.n_qubits, dtype=int), wires=kernel_instance.wires)
            return qml.expval(projector)

        def compute_similarities(x0, x1):
            qnode = qml.QNode(kernel_instance.circuit, kernel_instance.q_device)

            def _func(indices):
                b_x0, b_x1 = x0[indices[:, 0]], x1[indices[:, 1]]
                return qnode(b_x0, b_x1)

            gram = GramMatrix((x0.shape[0], x1.shape[0]), requires_grad=False)
            gram.apply_(_func, batch_size=kernel_instance.gram_batch_size, symmetrize=True)
            return gram.to_tensor()

        kernel_instance.ansatz = ansatz
        kernel_instance.circuit = circuit
        kernel_instance.compute_similarities = compute_similarities
        return kernel_instance

    def test_fit(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        assert kernel_instance.x_train_ is x_train
        assert isinstance(kernel_instance.bias_, torch.Tensor)
        assert isinstance(kernel_instance.data_scaling_, torch.Tensor)
        assert isinstance(kernel_instance.depth_, int)
        assert kernel_instance.is_fitted_

    def test_swap_test(self, kernel_instance):
        x_train = np.stack([np.linspace(0.0, 1.0, num=8) for _ in range(10)])
        kernel_instance.fit(x_train)
        kernel_matrix = kernel_instance(x_train)
        torch.testing.assert_close(kernel_matrix, torch.ones_like(kernel_matrix))

    def test_symmetric_kernel(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        kernel_matrix = kernel_instance(x_train)
        torch.testing.assert_close(kernel_matrix, kernel_matrix.T)

    def test_kernel_against_state_vector_kernel(self, kernel_instance, state_vector_kernel, x_train):
        kernel_instance.fit(x_train)
        kernel_instance.bias_.data = torch.zeros_like(kernel_instance.bias_.data)
        kernel_instance.data_scaling_.data = torch.ones_like(kernel_instance.data_scaling_.data)
        state_vector_kernel.fit(x_train)
        kernel_matrix = kernel_instance(x_train)
        sts_kernel_matrix = state_vector_kernel(x_train)
        torch.testing.assert_close(kernel_matrix, sts_kernel_matrix)

    def test_fit_with_preset_depth(self, kernel_instance, x_train):
        kernel_instance.depth_ = 2
        kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 2


class TestFermionicPQCKernelMiscellaneous:
    def test_invalid_entangling_method_raises(self):
        with pytest.raises(ValueError, match="Unknown entangling method"):
            FermionicPQCKernel(n_qubits=4, entangling_mth="bad_method")

    def test_ansatz_invalid_entangling_method_raises(self):
        import torch

        kernel = FermionicPQCKernel(n_qubits=4, entangling_mth="identity")
        kernel.entangling_mth = "bad_method"
        kernel.depth_ = 1
        kernel.bias_ = torch.zeros(4)
        kernel.data_scaling_ = torch.ones(4)
        with pytest.raises(ValueError, match="Unknown entangling method"):
            list(kernel.ansatz(np.zeros((1, 4))))
