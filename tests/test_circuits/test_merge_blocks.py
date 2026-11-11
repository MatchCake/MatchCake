import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake.circuits import merge_blocks
from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.operations.state_preparation import ProductState

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestMergeBlocks:
    @staticmethod
    def gates_block(params, wires):
        mc.operations.CompRxRx(params, wires=wires)
        mc.operations.CompRzRz(params, wires=wires)

    @staticmethod
    def next_block(params, wires):
        for left, right in zip(wires[:-1], wires[1:]):
            mc.operations.CompRxRx(params, wires=[left, right])
            mc.operations.CompRyRy(params, wires=[left, right])

    @staticmethod
    def random_product_state(wires, batch_size=None):
        shape = ([batch_size] if batch_size is not None else []) + [len(wires), 2]
        amplitudes = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        amplitudes = amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        return ProductState(amplitudes, wires=wires)

    @staticmethod
    def probs(qfunc, device_name, n_wires):
        @qml.qnode(qml.device(device_name, wires=n_wires))
        def circuit():
            qfunc()
            return qml.probs(wires=range(n_wires))

        return qml.math.to_numpy(circuit())

    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def assert_matches_qubit_simulation(self, merged_qfunc, reference_qfunc, n_wires):
        np.testing.assert_allclose(
            self.probs(merged_qfunc, "nif.qubit", n_wires),
            self.probs(reference_qfunc, "default.qubit", n_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_without_state_preps_keeps_gates(self):
        params_a, params_b, params_next = np.random.uniform(-np.pi, np.pi, size=(3, 2))

        def merged_qfunc():
            merge_blocks(
                lambda: self.gates_block(params_a, wires=[0, 1]),
                lambda: self.gates_block(params_b, wires=[2, 3]),
            )
            self.next_block(params_next, wires=[0, 1, 2, 3])

        def reference_qfunc():
            self.gates_block(params_a, wires=[0, 1])
            self.gates_block(params_b, wires=[2, 3])
            self.next_block(params_next, wires=[0, 1, 2, 3])

        self.assert_matches_qubit_simulation(merged_qfunc, reference_qfunc, 4)

    @pytest.mark.parametrize(
        "make_prep_b",
        [
            lambda: qml.BasisState(np.array([0, 1]), wires=[2, 3]),
            lambda: TestMergeBlocks.random_product_state([2, 3]),
            lambda: mc.operations.state_preparation.StatePrepFromGates(
                lambda wires: iter([qml.Hadamard(wires=wires[0]), qml.RY(0.3, wires=wires[1])]), wires=[2, 3]
            ),
        ],
    )
    def test_merges_state_preps_of_each_block(self, make_prep_b):
        params_a, params_b, params_next = np.random.uniform(-np.pi, np.pi, size=(3, 2))
        with qml.QueuingManager.stop_recording():
            prep_b = make_prep_b()

        def block_a():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            self.gates_block(params_a, wires=[0, 1])

        def block_b():
            qml.apply(prep_b)
            self.gates_block(params_b, wires=[2, 3])

        def merged_qfunc():
            merge_blocks(block_a, block_b)
            self.next_block(params_next, wires=[0, 1, 2, 3])

        def reference_qfunc():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            qml.apply(prep_b)
            self.gates_block(params_a, wires=[0, 1])
            self.gates_block(params_b, wires=[2, 3])
            self.next_block(params_next, wires=[0, 1, 2, 3])

        self.assert_matches_qubit_simulation(merged_qfunc, reference_qfunc, 4)

    def test_block_without_state_prep_starts_in_zero(self):
        params_a, params_b = np.random.uniform(-np.pi, np.pi, size=(2, 2))
        prep_a = self.random_product_state([0, 1])

        def block_a():
            qml.apply(prep_a)
            self.gates_block(params_a, wires=[0, 1])

        ops = merge_blocks(block_a, lambda: self.gates_block(params_b, wires=[2, 3]))

        np.testing.assert_equal(isinstance(ops[0], ProductState), True)
        np.testing.assert_equal(ops[0].wires.tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(ops[0].parameters[0][2:], [[1, 0], [1, 0]], atol=ATOL_MATRIX_COMPARISON)

    def test_contract_merges_gates_into_one_sptm(self):
        params_a, params_b, params_next = np.random.uniform(-np.pi, np.pi, size=(3, 2))

        def block_a():
            qml.BasisState(np.array([1, 1]), wires=[0, 1])
            self.gates_block(params_a, wires=[0, 1])

        def merged_qfunc():
            merge_blocks(block_a, lambda: self.gates_block(params_b, wires=[2, 3]), contract=True)
            self.next_block(params_next, wires=[0, 1, 2, 3])

        def reference_qfunc():
            qml.BasisState(np.array([1, 1]), wires=[0, 1])
            self.gates_block(params_a, wires=[0, 1])
            self.gates_block(params_b, wires=[2, 3])
            self.next_block(params_next, wires=[0, 1, 2, 3])

        with qml.QueuingManager.stop_recording():
            ops = merge_blocks(block_a, lambda: self.gates_block(params_b, wires=[2, 3]), contract=True)
        np.testing.assert_equal(len(ops), 2)
        np.testing.assert_equal(isinstance(ops[1], SingleParticleTransitionMatrixOperation), True)
        self.assert_matches_qubit_simulation(merged_qfunc, reference_qfunc, 4)

    def test_contract_with_state_preps_only(self):
        ops = merge_blocks(lambda: qml.BasisState(np.array([1, 0]), wires=[0, 1]), contract=True)

        np.testing.assert_equal(len(ops), 1)
        np.testing.assert_equal(isinstance(ops[0], ProductState), True)

    def test_extra_wires_cover_whole_device(self):
        params_a, params_next = np.random.uniform(-np.pi, np.pi, size=(2, 2))

        def block_a():
            qml.BasisState(np.array([0, 1]), wires=[0, 1])
            self.gates_block(params_a, wires=[0, 1])

        def merged_qfunc():
            merge_blocks(block_a, wires=range(4))
            self.next_block(params_next, wires=[0, 1, 2, 3])

        def reference_qfunc():
            qml.BasisState(np.array([0, 1]), wires=[0, 1])
            self.gates_block(params_a, wires=[0, 1])
            self.next_block(params_next, wires=[0, 1, 2, 3])

        self.assert_matches_qubit_simulation(merged_qfunc, reference_qfunc, 4)

    def test_batched_state_prep(self):
        batch_size = 3
        params_a, params_b = np.random.uniform(-np.pi, np.pi, size=(2, 2))
        prep_b = self.random_product_state([2, 3], batch_size=batch_size)

        def block_a():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            self.gates_block(params_a, wires=[0, 1])

        def block_b():
            qml.apply(prep_b)
            self.gates_block(params_b, wires=[2, 3])

        ops = merge_blocks(block_a, block_b)

        np.testing.assert_equal(ops[0].batch_size, batch_size)
        np.testing.assert_equal(qml.math.shape(ops[0].parameters[0]), (batch_size, 4, 2))
        np.testing.assert_allclose(ops[0].parameters[0][:, 2:], prep_b.parameters[0], atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(
            ops[0].parameters[0][:, :2],
            np.broadcast_to([[0, 1], [1, 0]], (batch_size, 2, 2)),
            atol=ATOL_MATRIX_COMPARISON,
        )

    def test_torch_state_prep_keeps_interface(self):
        amplitudes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
        ops = merge_blocks(
            lambda: ProductState(amplitudes, wires=[0, 1]),
            lambda: qml.BasisState(np.array([1, 0]), wires=[2, 3]),
        )

        np.testing.assert_equal(qml.math.get_interface(ops[0].parameters[0]), "torch")

    def test_queues_merged_operations_only_when_recording(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)

        def block_a():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            self.gates_block(params, wires=[0, 1])

        with qml.queuing.AnnotatedQueue() as queue:
            ops = merge_blocks(block_a, lambda: self.gates_block(params, wires=[2, 3]))

        np.testing.assert_equal(len(ops), 5)
        np.testing.assert_equal(len(queue.queue), len(ops))
        np.testing.assert_equal(all(queued is op for queued, op in zip(queue.queue, ops)), True)

    def test_no_blocks(self):
        np.testing.assert_equal(merge_blocks(), [])

    def test_identity_before_state_prep_is_allowed(self):
        def block_a():
            qml.Identity(wires=0)
            qml.BasisState(np.array([1, 0]), wires=[0, 1])

        ops = merge_blocks(block_a)

        np.testing.assert_equal(isinstance(ops[0], ProductState), True)

    def test_raises_on_shared_wires(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)
        with pytest.raises(ValueError, match="disjoint wires"):
            merge_blocks(
                lambda: self.gates_block(params, wires=[0, 1]),
                lambda: self.gates_block(params, wires=[1, 2]),
            )

    def test_raises_on_state_prep_after_gate(self):
        params = np.random.uniform(-np.pi, np.pi, size=2)

        def block_a():
            self.gates_block(params, wires=[0, 1])
            qml.BasisState(np.array([1, 0]), wires=[0, 1])

        with pytest.raises(ValueError, match="must come before the gates"):
            merge_blocks(block_a)

    def test_raises_on_non_product_state_prep(self):
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        with pytest.raises(ValueError, match="Only product state"):
            merge_blocks(lambda: qml.StatePrep(bell_state, wires=[0, 1]))

    def test_raises_on_overlapping_state_preps_in_block(self):
        def block_a():
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            qml.BasisState(np.array([1]), wires=[1])

        with pytest.raises(ValueError, match="State preparations must act on disjoint wires"):
            merge_blocks(block_a)

    def test_raises_on_different_batch_sizes(self):
        prep_a = self.random_product_state([0, 1], batch_size=2)
        prep_b = self.random_product_state([2, 3], batch_size=3)
        with pytest.raises(ValueError, match="same batch size"):
            merge_blocks(lambda: qml.apply(prep_a), lambda: qml.apply(prep_b))

    @pytest.mark.parametrize("contract", [False, True])
    def test_gradcheck_gate_params(self, contract):
        params_a, params_b = (
            torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), dtype=torch.float64, requires_grad=True)
            for _ in range(2)
        )
        params_next = torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), dtype=torch.float64)

        def expval(block_params_a, block_params_b):
            def block_a():
                qml.BasisState(np.array([1, 0]), wires=[0, 1])
                self.gates_block(block_params_a, wires=[0, 1])

            @qml.qnode(qml.device("nif.qubit", wires=4), interface="torch")
            def circuit():
                merge_blocks(block_a, lambda: self.gates_block(block_params_b, wires=[2, 3]), contract=contract)
                self.next_block(params_next, wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(1) @ qml.Z(2))

            return circuit()

        torch.autograd.gradcheck(expval, (params_a, params_b))

    @pytest.mark.parametrize("contract", [False, True])
    def test_gate_params_gradient_matches_qubit_simulation(self, contract):
        initial_params = np.random.uniform(-np.pi, np.pi, size=(3, 2))
        gradients = {}
        for device_name in ["nif.qubit", "default.qubit"]:
            params_a, params_b, params_next = (
                torch.tensor(initial_params[index], dtype=torch.float64, requires_grad=True) for index in range(3)
            )

            def block_a():
                qml.BasisState(np.array([1, 0]), wires=[0, 1])
                self.gates_block(params_a, wires=[0, 1])

            def block_b():
                qml.BasisState(np.array([0, 1]), wires=[2, 3])
                self.gates_block(params_b, wires=[2, 3])

            @qml.qnode(qml.device(device_name, wires=4), interface="torch")
            def circuit():
                if device_name == "nif.qubit":
                    merge_blocks(block_a, block_b, contract=contract)
                else:
                    qml.BasisState(np.array([1, 0, 0, 1]), wires=[0, 1, 2, 3])
                    self.gates_block(params_a, wires=[0, 1])
                    self.gates_block(params_b, wires=[2, 3])
                self.next_block(params_next, wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(1) @ qml.Z(2))

            circuit().backward()
            gradients[device_name] = [params.grad for params in (params_a, params_b, params_next)]

        for nif_gradient, qubit_gradient in zip(gradients["nif.qubit"], gradients["default.qubit"]):
            torch.testing.assert_close(
                nif_gradient, qubit_gradient, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    @pytest.mark.parametrize("contract", [False, True])
    def test_gradcheck_state_prep_amplitudes(self, contract):
        amplitudes = torch.tensor(self.random_product_state([0, 1]).parameters[0], requires_grad=True)
        params_b, params_next = (
            torch.tensor(np.random.uniform(-np.pi, np.pi, size=2), dtype=torch.float64) for _ in range(2)
        )

        def expval(state_amplitudes):
            @qml.qnode(qml.device("nif.qubit", wires=4), interface="torch")
            def circuit():
                merge_blocks(
                    lambda: ProductState(state_amplitudes, wires=[0, 1], validate_norm=False),
                    lambda: self.gates_block(params_b, wires=[2, 3]),
                    contract=contract,
                )
                self.next_block(params_next, wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(1) @ qml.Z(2))

            return circuit()

        torch.autograd.gradcheck(expval, (amplitudes,))
