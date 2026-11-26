import copy
import warnings

import numpy as np
import pennylane as qml
import pytest
import torch.nn
from sklearn.base import clone

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
        with pytest.warns(UserWarning, match="features"):
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
        with pytest.warns(UserWarning, match="features"):
            kernel_instance.fit(np.concatenate([x_train] * 4, axis=-1))
        assert kernel_instance.depth_ == 2
        kernel_instance.depth_ = None
        kernel_instance.fit(np.concatenate([x_train] * 4, axis=-1))
        assert kernel_instance.depth_ == 32 // kernel_instance.n_qubits

    def test_refit_with_another_number_of_features(self, kernel_instance, x_train, x_test):
        assert kernel_instance.depth_ is None
        kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 8 // kernel_instance.n_qubits
        wide_x_train = np.concatenate([x_train, x_train[::-1] + 0.3], axis=-1)
        wide_x_test = np.concatenate([x_test, x_test[::-1] + 0.1], axis=-1)
        kernel_instance.fit(wide_x_train)
        assert kernel_instance.depth_ == 16 // kernel_instance.n_qubits
        with torch.no_grad():
            kernel = kernel_instance(wide_x_test, wide_x_train)
            other_wide_x_train = wide_x_train.copy()
            other_wide_x_train[:, 8:] += np.random.default_rng(TEST_SEED).random((len(x_train), 8))
            other_kernel = kernel_instance(wide_x_test, other_wide_x_train)
        if kernel_instance.rotations != "Z":
            assert not torch.allclose(kernel, other_kernel)
        self.assert_transform_is_up_to_date(kernel_instance, wide_x_test)
        kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 8 // kernel_instance.n_qubits
        self.assert_transform_is_up_to_date(kernel_instance, x_test)

    def test_depth_follows_the_flattened_number_of_features(self, kernel_instance, x_train):
        kernel_instance.fit(np.stack([x_train, x_train + 0.1], axis=1))
        assert kernel_instance.depth_ == 16 // kernel_instance.n_qubits
        assert kernel_instance.bias_.shape == (16,)

    def test_float64_kernel_is_not_rounded_to_float32(self, n_qubits, rotations, entangling_mth, x_test):
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits,
            rotations=rotations,
            entangling_mth=entangling_mth,
            r_dtype=torch.float64,
            c_dtype=torch.complex128,
        ).fit(x_test)
        with torch.no_grad():
            kernel = kernel_instance(x_test, x_test[::-1].copy())
        assert kernel.dtype == torch.float64
        if rotations != "Z":
            assert not torch.equal(kernel, kernel.to(torch.float32).to(torch.float64))

    def test_depth_of_a_kernel_pickled_without_the_property(self, kernel_instance, x_train):
        del kernel_instance.__dict__["_depth"]
        del kernel_instance.__dict__["depth"]
        kernel_instance.__dict__["depth_"] = 2
        assert kernel_instance.depth_ == 2
        kernel_instance.fit(x_train)
        assert "depth_" not in kernel_instance.__dict__
        assert kernel_instance.depth_ == 8 // kernel_instance.n_qubits

    def test_parameters_of_a_kernel_pickled_without_the_property(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        kernel_instance.__dict__["depth_"] = kernel_instance.__dict__.pop("_depth")
        del kernel_instance.__dict__["depth"]
        assert kernel_instance.get_params()["depth"] is None
        assert clone(kernel_instance).depth_ is None
        kernel_instance.set_params(depth=3)
        assert kernel_instance.fit(x_train).depth_ == 3

    def test_depth_set_by_hand_is_in_use_before_fit(self, kernel_instance):
        kernel_instance.depth_ = 3
        assert kernel_instance.depth_ == 3
        kernel_instance.depth_ = None
        assert kernel_instance.depth_ is None

    def test_constructor_stores_depth_as_given(self, n_qubits, rotations, entangling_mth):
        depth = np.int64(3)
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth, depth=depth
        )
        assert kernel_instance.depth is depth
        assert isinstance(kernel_instance.depth_, int)

    def test_depth_argument(self, n_qubits, rotations, entangling_mth, x_train):
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth, depth=3
        )
        assert kernel_instance.depth_ == 3
        assert kernel_instance.get_params()["depth"] == 3
        kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 3
        with pytest.warns(UserWarning, match="features"):
            cloned = clone(kernel_instance).fit(np.concatenate([x_train] * 4, axis=-1))
        assert cloned.depth_ == 3
        kernel_instance.set_params(depth=None)
        kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 8 // n_qubits
        kernel_instance.set_params(depth=2)
        kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 2

    def test_clone_keeps_a_depth_set_by_hand(self, kernel_instance, x_train):
        kernel_instance.depth_ = np.int64(2)
        assert kernel_instance.depth == 2
        assert isinstance(kernel_instance.depth, int)
        cloned = clone(kernel_instance)
        assert cloned.depth_ == 2
        with pytest.warns(UserWarning, match="features"):
            cloned.fit(np.concatenate([x_train] * 4, axis=-1))
        assert cloned.depth_ == 2
        kernel_instance.depth_ = None
        assert clone(kernel_instance).depth_ is None

    def test_depth_set_to_none_on_a_fitted_kernel(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        expected = kernel_instance.transform(x_train)
        kernel_instance.depth_ = 5
        assert kernel_instance.depth_ == 5
        kernel_instance.depth_ = None
        assert kernel_instance.depth is None
        assert kernel_instance.depth_ == 8 // kernel_instance.n_qubits
        np.testing.assert_allclose(
            kernel_instance.transform(x_train), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("depth", [1, 2])
    def test_depth_too_small_for_the_features_warns(self, n_qubits, rotations, entangling_mth, x_train, depth):
        x_wide = np.concatenate([x_train] * 4, axis=-1)
        kernel_arguments = dict(n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth)
        kernel_instance = FermionicPQCKernel(depth=depth, **kernel_arguments)
        with pytest.warns(UserWarning, match=f"only the first {depth * n_qubits} of the 32 features") as record:
            kernel_instance.fit(x_wide)
        assert record[0].filename == __file__
        assert kernel_instance.depth_ == depth
        kernel_instance = FermionicPQCKernel(**kernel_arguments).fit(x_wide)
        with pytest.warns(UserWarning, match=f"A depth of {32 // n_qubits} is needed") as record:
            kernel_instance.depth_ = depth
        assert record[0].filename == __file__
        assert kernel_instance.depth_ == depth

    def test_depth_setter_changes_nothing_when_the_warning_is_an_error(self, kernel_instance, x_train):
        kernel_instance.fit(np.concatenate([x_train] * 4, axis=-1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="features"):
                kernel_instance.depth_ = 1
        assert kernel_instance.depth is None
        assert kernel_instance.depth_ == 32 // kernel_instance.n_qubits

    def test_depth_set_to_none_follows_the_training_data(self, kernel_instance, x_train):
        kernel_instance.fit(np.concatenate([x_train] * 4, axis=-1))
        kernel_instance.bias_ = torch.zeros(1)
        kernel_instance.depth_ = None
        assert kernel_instance.depth_ == 32 // kernel_instance.n_qubits

    def test_depth_large_enough_does_not_warn(self, n_qubits, rotations, entangling_mth, x_train, recwarn):
        x_wide = np.concatenate([x_train] * 2, axis=-1)
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth, depth=16 // n_qubits
        )
        kernel_instance.fit(x_wide)
        kernel_instance.depth_ = 16 // n_qubits + 1
        kernel_instance.fit(x_wide)
        assert not [warning for warning in recwarn if "features" in str(warning.message)]

    def test_depth_above_the_maximum_raises(self, n_qubits, rotations, entangling_mth, x_train):
        kernel_arguments = dict(n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth)
        assert FermionicPQCKernel(depth=FermionicPQCKernel.MAX_DEPTH, **kernel_arguments).depth_ == 10_000
        with pytest.raises(ValueError, match="at most 10000"):
            FermionicPQCKernel(depth=FermionicPQCKernel.MAX_DEPTH + 1, **kernel_arguments)
        with pytest.raises(ValueError, match="at most 10000"):
            FermionicPQCKernel(depth=10**30, **kernel_arguments)
        kernel_instance = FermionicPQCKernel(**kernel_arguments).fit(x_train)
        with pytest.raises(ValueError, match="at most 10000"):
            kernel_instance.depth_ = FermionicPQCKernel.MAX_DEPTH + 1
        kernel_instance.set_params(depth=FermionicPQCKernel.MAX_DEPTH + 1)
        with pytest.raises(ValueError, match="at most 10000"):
            kernel_instance.fit(x_train)
        assert kernel_instance.depth_ == 8 // n_qubits

    def test_maximum_depth_of_a_subclass(self, n_qubits, rotations, entangling_mth):
        class ShallowKernel(FermionicPQCKernel):
            MAX_DEPTH = 3

        kernel_arguments = dict(n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth)
        assert ShallowKernel(depth=3, **kernel_arguments).depth_ == 3
        with pytest.raises(ValueError, match="at most 3"):
            ShallowKernel(depth=4, **kernel_arguments)

    def test_failed_fit_keeps_the_fitted_state(self, kernel_instance, x_train):
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        state = {name: value for name, value in kernel_instance.__dict__.items() if not name.startswith("_")}
        parameters = dict(kernel_instance.named_parameters())
        depth = kernel_instance.depth_
        expected = kernel_instance.transform(x_train)
        with pytest.raises(Exception):
            kernel_instance.fit(np.linspace(0.0, 1.0, num=10))
        assert kernel_instance.x_train_ is x_train
        assert kernel_instance.is_fitted_
        assert kernel_instance.depth_ == depth
        assert set(state) == {name for name in kernel_instance.__dict__ if not name.startswith("_")}
        for name, value in dict(kernel_instance.named_parameters()).items():
            assert value is parameters[name]
        np.testing.assert_allclose(
            kernel_instance.transform(x_train), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("depth", [0, -1, 1.5, "2", True])
    def test_invalid_depth_raises(self, n_qubits, rotations, entangling_mth, x_train, depth):
        kernel_arguments = dict(n_qubits=n_qubits, rotations=rotations, entangling_mth=entangling_mth)
        with pytest.raises(ValueError, match="positive integer"):
            FermionicPQCKernel(depth=depth, **kernel_arguments)
        kernel_instance = FermionicPQCKernel(**kernel_arguments)
        with pytest.raises(ValueError, match="positive integer"):
            kernel_instance.depth_ = depth
        assert kernel_instance.depth_ is None
        kernel_instance.fit(x_train)
        kernel_instance.set_params(depth=depth)
        with pytest.raises(ValueError, match="positive integer"):
            kernel_instance.fit(x_train[:5])
        assert kernel_instance.x_train_ is x_train
        assert kernel_instance.depth_ == 8 // n_qubits
        with pytest.raises(ValueError, match="positive integer"):
            kernel_instance.fit(np.concatenate([x_train] * 2, axis=-1))
        assert kernel_instance.bias_.shape == (8,)
        assert kernel_instance.data_scaling_.shape == (8,)

    @pytest.mark.parametrize("alignment_dtype", [torch.float32, torch.float64])
    def test_kernel_dtype_and_alignment(self, n_qubits, rotations, entangling_mth, x_train, alignment_dtype):
        c_dtype = torch.complex64 if alignment_dtype == torch.float32 else torch.complex128
        kernel_instance = FermionicPQCKernel(
            n_qubits=n_qubits,
            rotations=rotations,
            entangling_mth=entangling_mth,
            r_dtype=alignment_dtype,
            c_dtype=c_dtype,
            alignment=True,
            alignment_iterations=2,
        )
        kernel_instance.fit(x_train, np.arange(len(x_train)) % 2)
        assert len(kernel_instance.opt_.history) == 2
        assert kernel_instance.transform(torch.from_numpy(x_train)).dtype == alignment_dtype
        kernel_instance.train()
        assert kernel_instance(x_train).dtype == alignment_dtype


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
