import copy

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

from ...configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED


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

    @pytest.fixture
    def x_test(self):
        return np.random.default_rng(TEST_SEED).random((4, 8))

    def test_cross_kernel_against_joint_gram_matrix(self, kernel_instance, x_train, x_test):
        kernel_instance.fit(x_train)
        n_train = len(x_train)
        with torch.no_grad():
            joint_gram = kernel_instance(np.concatenate([x_train, x_test], axis=0))
            cross_kernel = kernel_instance(x_train, x_test)
            transposed_cross_kernel = kernel_instance(x_test, x_train)
            shuffled_cross_kernel = kernel_instance(x_train, x_train[::-1].copy())
        torch.testing.assert_close(
            cross_kernel, joint_gram[:n_train, n_train:], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            transposed_cross_kernel,
            joint_gram[n_train:, :n_train],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        torch.testing.assert_close(
            shuffled_cross_kernel,
            torch.flip(joint_gram[:n_train, :n_train], dims=(1,)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_cross_kernel_of_single_samples(self, kernel_instance, x_test):
        kernel_instance.fit(x_test)
        with torch.no_grad():
            gram = kernel_instance(x_test[:2])
            single = kernel_instance(x_test[0:1], x_test[1:2])
        assert single.shape == (1, 1)
        torch.testing.assert_close(single[0, 0], gram[0, 1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_transform_against_joint_gram_matrix(self, kernel_instance, x_train, x_test):
        kernel_instance.fit(x_train)
        n_train = len(x_train)
        with torch.no_grad():
            joint_gram = kernel_instance(np.concatenate([x_train, x_test], axis=0))
        transformed = kernel_instance.transform(torch.from_numpy(x_test))
        torch.testing.assert_close(
            transformed, joint_gram[n_train:, :n_train], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_transform_after_refit(self, kernel_instance, x_train, x_test):
        x_test_tensor = torch.from_numpy(x_test)
        for x_fit in [x_train, x_train[:5] + 0.25, x_train]:
            kernel_instance.fit(x_fit)
            transformed = kernel_instance.transform(x_test_tensor)
            with torch.no_grad():
                expected = kernel_instance(x_test, x_fit)
            torch.testing.assert_close(transformed, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def assert_transform_is_up_to_date(self, kernel_instance, x_test):
        transformed = kernel_instance.transform(torch.from_numpy(x_test))
        with torch.no_grad():
            expected = kernel_instance(x_test, kernel_instance.x_train_)
        torch.testing.assert_close(transformed, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_transform_follows_the_state_of_the_kernel(self, kernel_instance, x_train, x_test):
        kernel_instance.fit(x_train)
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        kernel_instance.bias_.data = kernel_instance.bias_.data + 0.3
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        with torch.no_grad():
            kernel_instance.data_scaling_.mul_(0.5)
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        other = FermionicPQCKernel(
            n_qubits=kernel_instance.n_qubits,
            rotations=kernel_instance.rotations,
            entangling_mth=kernel_instance.entangling_mth,
            random_state=7,
        ).fit(x_train)
        kernel_instance.load_state_dict(other.state_dict())
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        kernel_instance.x_train_ = x_train[:6] + 0.1
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        kernel_instance.entangling_mth = "identity" if kernel_instance.entangling_mth != "identity" else "fswap"
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        kernel_instance.rotations = "X,Z" if kernel_instance.rotations != "X,Z" else "Y,Z"
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        kernel_instance.x_train_ = np.concatenate([x_train, x_train[:, ::-1] + 0.2], axis=-1)
        kernel_instance.bias_ = torch.cat([kernel_instance.bias_.detach(), kernel_instance.bias_.detach() + 0.1])
        kernel_instance.data_scaling_ = torch.cat([kernel_instance.data_scaling_.detach()] * 2)
        x_test = np.concatenate([x_test, x_test[:, ::-1]], axis=-1)
        kernel_instance.depth_ = 1
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        kernel_instance.depth_ = int(np.ceil(x_test.shape[-1] / kernel_instance.n_qubits))
        self.assert_transform_is_up_to_date(kernel_instance, x_test)

    def test_transform_follows_in_place_changes_of_tensor_x_train(self, kernel_instance, x_train, x_test):
        x_train_tensor = torch.from_numpy(x_train).clone()
        kernel_instance.fit(x_train_tensor)
        self.assert_transform_is_up_to_date(kernel_instance, x_test)
        x_train_tensor.add_(0.3)
        self.assert_transform_is_up_to_date(kernel_instance, x_test)

    def test_train_sptms_carry_gradients_after_a_transform(self, kernel_instance, x_train, x_test):
        kernel_instance.fit(x_train)
        kernel_instance.transform(torch.from_numpy(x_test))
        kernel_instance.train()
        gradients = []
        for x1_train in [True, False, True]:
            kernel_instance.zero_grad()
            kernel = kernel_instance(x_test, kernel_instance.x_train_, x1_train=x1_train)
            kernel.pow(2).sum().backward()
            gradients.append(kernel_instance.bias_.grad.detach().clone())
        torch.testing.assert_close(gradients[0], gradients[1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        torch.testing.assert_close(gradients[2], gradients[1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        copy.deepcopy(kernel_instance)

    def test_fit_draws_the_bias_from_random_state(self, n_qubits, rotations, entangling_mth, x_train):
        for random_state in [0, 3]:
            kernel = FermionicPQCKernel(
                n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth, random_state=random_state
            ).fit(x_train)
            expected = np.random.RandomState(random_state).random(x_train.shape[-1])
            torch.testing.assert_close(kernel.bias_.detach(), torch.from_numpy(expected).to(kernel.R_DTYPE))

    def test_fit_on_zero_samples_keeps_the_fitted_state(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        with pytest.raises(ValueError, match="zero samples"):
            kernel_instance.fit(np.empty((0, 12)))
        assert kernel_instance.bias_.shape == (x_train.shape[-1],)
        assert kernel_instance.x_train_ is x_train

    def test_refit_on_same_data_gives_same_kernel(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        with torch.no_grad():
            first_kernel = kernel_instance(x_train)
        first_bias = kernel_instance.bias_.detach().clone()
        kernel_instance.fit(x_train)
        with torch.no_grad():
            second_kernel = kernel_instance(x_train)
        torch.testing.assert_close(kernel_instance.bias_, first_bias, atol=0.0, rtol=0.0)
        torch.testing.assert_close(second_kernel, first_kernel, atol=0.0, rtol=0.0)

    def test_parameters_are_registered(self, kernel_instance, x_train):
        assert kernel_instance.bias_ is None
        assert kernel_instance.data_scaling_ is None
        kernel_instance.fit(x_train)
        parameters = dict(kernel_instance.named_parameters())
        for name in ["bias_", "data_scaling_"]:
            assert isinstance(parameters[name], torch.nn.Parameter)
            assert parameters[name].is_leaf
            assert parameters[name].requires_grad
            assert parameters[name].dtype == kernel_instance.R_DTYPE
        torch.testing.assert_close(
            kernel_instance.data_scaling_, torch.full_like(kernel_instance.data_scaling_, torch.pi)
        )

    def test_plain_tensors_are_wrapped_into_parameters(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        kernel_instance.bias_ = torch.zeros(x_train.shape[-1])
        kernel_instance.data_scaling_ = torch.ones(x_train.shape[-1]) * torch.rand(1).requires_grad_()
        parameters = dict(kernel_instance.named_parameters())
        for name in ["bias_", "data_scaling_"]:
            assert isinstance(parameters[name], torch.nn.Parameter)
            assert parameters[name].is_leaf
        torch.testing.assert_close(kernel_instance.bias_, torch.zeros(x_train.shape[-1]))
        assert kernel_instance.bias_.requires_grad
        kernel_instance.freeze()
        kernel_instance.bias_ = torch.ones(x_train.shape[-1])
        assert not kernel_instance.bias_.requires_grad
        kernel_instance.data_scaling_ = torch.ones(x_train.shape[-1], dtype=torch.long)
        assert isinstance(kernel_instance.data_scaling_, torch.nn.Parameter)
        assert not kernel_instance.data_scaling_.requires_grad

    def test_alignment_with_several_iterations(self, kernel_instance, x_train):
        kernel_instance.alignment = True
        kernel_instance.alignment_iterations = 3
        kernel_instance.alignment_early_stopping_patience = 100
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        assert kernel_instance.is_fitted_
        assert len(kernel_instance.opt_.history) == 3
        assert all(np.isfinite(kernel_instance.opt_.history))
        assert torch.all(torch.isfinite(kernel_instance.bias_))

    def test_alignment_trains_the_parameters(self, n_qubits, rotations, entangling_mth, x_train):
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits,
            rotations=rotations,
            entangling_mth=entangling_mth,
            alignment=True,
            alignment_iterations=4,
            alignment_learning_rate=5e-2,
            alignment_early_stopping_patience=100,
        )
        x_train = np.random.default_rng(TEST_SEED).random(x_train.shape)
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        initial_scaling = torch.full_like(kernel_instance.data_scaling_, torch.pi)
        assert not torch.allclose(kernel_instance.data_scaling_, initial_scaling)
        assert max(kernel_instance.opt_.history) > kernel_instance.opt_.history[0]

    def test_deepcopy_of_fitted_kernel(self, kernel_instance, x_train, x_test):
        kernel_instance.fit(x_train)
        expected = kernel_instance.transform(torch.from_numpy(x_test))
        kernel_instance.train()
        kernel_instance(x_train)
        copied = copy.deepcopy(kernel_instance)
        torch.testing.assert_close(copied.transform(torch.from_numpy(x_test)), expected, atol=0.0, rtol=0.0)
        assert copied.bias_ is not kernel_instance.bias_

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
