import numpy as np
import pennylane as qml
import pytest

import matchcake as mc
from matchcake.circuits import (
    RandomMatchgateOperationsGenerator,
    RandomSptmOperationsGenerator,
    merge_generators,
)
from matchcake.operations.state_preparation import ProductState

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)


class TestMergeGenerators:
    @staticmethod
    def gates(wires, seed):
        params = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=(2, 2))
        yield mc.operations.CompRxRx(params[0], wires=wires)
        yield mc.operations.CompRzRz(params[1], wires=wires)

    @staticmethod
    def nif_probs(op_iterator, n_wires):
        device = mc.NIFDevice(wires=n_wires)
        return qml.math.to_numpy(device.execute_generator(op_iterator, output_type="probs", reset=True))

    @staticmethod
    def qubit_probs(ops, n_wires):
        @qml.qnode(qml.device("default.qubit", wires=n_wires))
        def circuit():
            for op in ops:
                qml.apply(op)
            return qml.probs(wires=range(n_wires))

        return qml.math.to_numpy(circuit())

    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_generators_match_qubit_simulation(self, seed):
        left = RandomMatchgateOperationsGenerator(wires=[0, 1, 2], n_ops=6, seed=seed)
        right = RandomMatchgateOperationsGenerator(wires=[3, 4], n_ops=4, seed=seed + 10)
        left_ops, right_ops = left.tolist(), right.tolist()
        initial_state = np.concatenate([left_ops[0].parameters[0], right_ops[0].parameters[0]])
        reference_ops = [qml.BasisState(initial_state, wires=range(5))] + left_ops[1:] + right_ops[1:]

        np.testing.assert_allclose(
            self.nif_probs(merge_generators(left, right), 5),
            self.qubit_probs(reference_ops, 5),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_sptm_generators_on_nif_device(self):
        left = RandomSptmOperationsGenerator(wires=[0, 1, 2], seed=0)
        right = RandomSptmOperationsGenerator(wires=[3, 4, 5], seed=1)

        probs = self.nif_probs(merge_generators(left, right), 6)

        np.testing.assert_allclose(probs.sum(), 1.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_equal(probs.shape, (2**6,))

    def test_yields_merged_state_prep_first_then_gates_in_order(self):
        left = [qml.BasisState(np.array([1, 0]), wires=[0, 1]), *self.gates([0, 1], seed=0)]
        right = [ProductState.from_basis_state([0, 1], wires=[2, 3]), *self.gates([2, 3], seed=1)]

        ops = list(merge_generators(iter(left), iter(right)))

        np.testing.assert_equal(isinstance(ops[0], ProductState), True)
        np.testing.assert_equal(ops[0].wires.tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(
            ops[0].parameters[0],
            [[0, 1], [1, 0], [1, 0], [0, 1]],
            atol=ATOL_MATRIX_COMPARISON,
        )
        np.testing.assert_equal(ops[1:], left[1:] + right[1:])

    def test_without_state_preps_yields_only_gates(self):
        left, right = list(self.gates([0, 1], seed=0)), list(self.gates([2, 3], seed=1))

        ops = list(merge_generators(left, right))

        np.testing.assert_equal(ops, left + right)

    def test_wires_extend_state_prep_to_generators_without_one(self):
        left = [qml.BasisState(np.array([1, 1]), wires=[0, 1]), *self.gates([0, 1], seed=0)]
        right = list(self.gates([2, 3], seed=1))
        reference_ops = [qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))] + left[1:] + right

        ops = list(merge_generators(left, right, wires=[2, 3]))

        np.testing.assert_equal(ops[0].wires.tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(
            self.nif_probs(iter(ops), 4),
            self.qubit_probs(reference_ops, 4),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_generator_wires_attribute_extends_state_prep(self):
        class GatesOnly:
            wires = [0, 1]

            def __iter__(self):
                return TestMergeGenerators.gates(self.wires, seed=0)

        right = [qml.BasisState(np.array([1, 0]), wires=[2, 3]), *self.gates([2, 3], seed=1)]

        ops = list(merge_generators(GatesOnly(), right))

        np.testing.assert_equal(ops[0].wires.tolist(), [2, 3, 0, 1])
        np.testing.assert_allclose(ops[0].parameters[0], [[0, 1], [1, 0], [1, 0], [1, 0]], atol=ATOL_MATRIX_COMPARISON)

    def test_is_lazy(self):
        consumed = []

        def generator(wires, seed):
            yield qml.BasisState(np.array([1, 0]), wires=wires)
            for op in self.gates(wires, seed):
                consumed.append(op)
                yield op

        merged = merge_generators(generator([0, 1], seed=0), generator([2, 3], seed=1))
        np.testing.assert_equal(len(consumed), 0)

        first_op = next(merged)

        np.testing.assert_equal(isinstance(first_op, ProductState), True)
        np.testing.assert_equal(len(consumed), 2)
        list(merged)
        np.testing.assert_equal(len(consumed), 4)

    def test_leading_identities_are_kept(self):
        identity = qml.Identity(wires=0)
        left = [identity, qml.BasisState(np.array([1, 0]), wires=[0, 1]), *self.gates([0, 1], seed=0)]

        ops = list(merge_generators(left))

        np.testing.assert_equal(isinstance(ops[0], ProductState), True)
        np.testing.assert_equal(ops[1], identity)
        np.testing.assert_equal(ops[2:], left[2:])

    def test_empty_generators(self):
        np.testing.assert_equal(list(merge_generators([], [])), [])

    def test_can_be_consumed_inside_qnode(self):
        left = [qml.BasisState(np.array([1, 0]), wires=[0, 1]), *self.gates([0, 1], seed=0)]
        right = list(self.gates([2, 3], seed=1))

        def qfunc():
            for op in merge_generators(left, right, wires=[2, 3]):
                qml.apply(op)

        tape = qml.tape.make_qscript(qfunc)()

        np.testing.assert_equal(len(tape.operations), 1 + len(left[1:]) + len(right))

    def test_generator_ops_are_not_queued(self):
        def generator():
            yield qml.BasisState(np.array([1, 0]), wires=[0, 1])
            yield mc.operations.CompRxRx(np.array([0.1, 0.2]), wires=[0, 1])

        def qfunc():
            list(merge_generators(generator()))

        np.testing.assert_equal(qml.tape.make_qscript(qfunc)().operations, [])

    def test_shared_state_prep_wires_raise(self):
        left = [qml.BasisState(np.array([1, 0]), wires=[0, 1])]
        right = [qml.BasisState(np.array([1, 0]), wires=[1, 2])]

        with pytest.raises(ValueError, match="disjoint wires"):
            list(merge_generators(left, right))

    def test_shared_gate_wires_raise(self):
        left = list(self.gates([0, 1], seed=0))
        right = list(self.gates([1, 2], seed=1))

        with pytest.raises(ValueError, match="generators 0 and 1 share wire 1"):
            list(merge_generators(left, right))

    def test_state_prep_after_gate_raises(self):
        left = [*self.gates([0, 1], seed=0), qml.BasisState(np.array([1, 0]), wires=[0, 1])]

        with pytest.raises(ValueError, match="must come before the gates"):
            list(merge_generators(left))

    def test_non_product_state_prep_raises(self):
        left = [qml.StatePrep(np.array([1, 0, 0, 1]) / np.sqrt(2), wires=[0, 1])]

        with pytest.raises(ValueError, match="Only product state preparations"):
            list(merge_generators(left))

    def test_mismatched_batch_sizes_raise(self):
        left = [ProductState(np.tile([[1.0, 0.0], [0.0, 1.0]], (2, 1, 1)), wires=[0, 1])]
        right = [ProductState(np.tile([[1.0, 0.0], [0.0, 1.0]], (3, 1, 1)), wires=[2, 3])]

        with pytest.raises(ValueError, match="same batch size"):
            list(merge_generators(left, right))
