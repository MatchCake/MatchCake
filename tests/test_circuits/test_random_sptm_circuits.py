import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake as mc
from matchcake.circuits.random_sptm_circuits import (
    RandomSptmHaarOperationsGenerator,
    RandomSptmOperationsGenerator,
    random_sptm_operations_generator,
)
from matchcake.operations import SptmCompRxRx, SptmCompRyRy, SptmCompRzRz, SptmCompZX, SptmIdentity
from matchcake.operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)

from ..configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED, set_seed


class TestRandomSptmOperationsGenerator:
    @staticmethod
    def sptm_matrices(ops):
        return [qml.math.to_numpy(op.sptm_matrix()) for op in ops if not isinstance(op, qml.BasisState)]

    @staticmethod
    def wire_pairs(ops):
        return {tuple(op.wires.tolist()) for op in ops if not isinstance(op, qml.BasisState)}

    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_generator_yields_requested_number_of_ops(self):
        ops = list(random_sptm_operations_generator(n_ops=4, wires=3, seed=TEST_SEED))
        assert len(ops) == 4

    def test_generator_use_cuda_calls_to_cuda(self, monkeypatch):
        # ``to_cuda`` itself is GPU-only (and pragma-excluded); stub it so the
        # use_cuda=True call site is exercised on a CPU-only machine.
        calls = {"n": 0}

        def fake_to_cuda(self):
            calls["n"] += 1
            return self

        monkeypatch.setattr(SingleParticleTransitionMatrixOperation, "to_cuda", fake_to_cuda)
        ops = list(random_sptm_operations_generator(n_ops=2, wires=3, use_cuda=True, seed=TEST_SEED))
        assert len(ops) == 2
        assert calls["n"] == 2

    def test_generator_ops_of_same_type_have_different_params(self):
        ops = list(random_sptm_operations_generator(n_ops=6, wires=3, op_types=(SptmCompRxRx,), seed=TEST_SEED))
        matrices = self.sptm_matrices(ops)

        for index, matrix in enumerate(matrices[1:], start=1):
            assert not np.allclose(matrix, matrices[0]), f"Operation {index} has the parameters of operation 0."

    def test_generator_same_seed_gives_same_circuit(self):
        first = list(random_sptm_operations_generator(n_ops=8, wires=4, seed=TEST_SEED))
        second = list(random_sptm_operations_generator(n_ops=8, wires=4, seed=TEST_SEED))

        assert [type(op) for op in first] == [type(op) for op in second]
        assert [op.wires for op in first] == [op.wires for op in second]
        for matrix, other in zip(self.sptm_matrices(first), self.sptm_matrices(second)):
            np.testing.assert_allclose(matrix, other, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_generator_stays_on_neighbouring_pairs_of_wires(self):
        ops = list(random_sptm_operations_generator(n_ops=60, wires=[6, 0, 1, 5, 2], seed=TEST_SEED))

        assert self.wire_pairs(ops) == {(0, 1), (1, 2), (5, 6)}

    def test_generator_without_neighbouring_wires_raises(self):
        with pytest.raises(ValueError, match="neighbouring wires"):
            list(random_sptm_operations_generator(n_ops=2, wires=[0, 2, 5], seed=TEST_SEED))

    def test_class_yields_basis_state_then_ops_on_neighbouring_pairs(self):
        ops = list(RandomSptmOperationsGenerator(wires=[3, 4, 6, 7], n_ops=40, seed=TEST_SEED))

        assert isinstance(ops[0], qml.BasisState)
        assert len(ops) == 41
        assert self.wire_pairs(ops) == {(3, 4), (6, 7)}

    def test_class_ops_of_same_type_have_different_params(self):
        ops = list(RandomSptmOperationsGenerator(wires=3, n_ops=6, op_types=(SptmCompRxRx,), seed=TEST_SEED))
        matrices = self.sptm_matrices(ops)

        for matrix in matrices[1:]:
            assert not np.allclose(matrix, matrices[0])

    def test_haar_default_n_ops(self):
        generator = RandomSptmHaarOperationsGenerator(wires=4, seed=TEST_SEED)
        ops = list(generator)

        assert generator.n_ops == 3 * 4 * 3
        assert isinstance(ops[0], qml.BasisState)
        assert isinstance(ops[1], SptmIdentity)
        assert len(ops) == 2 + generator.n_ops

    @pytest.mark.parametrize("n_ops", [1, 2, 3, 4, 10, 13])
    @pytest.mark.parametrize("add_swap_noise", [True, False])
    def test_haar_yields_exactly_n_ops(self, n_ops, add_swap_noise):
        generator = RandomSptmHaarOperationsGenerator(wires=4, n_ops=n_ops, add_swap_noise=add_swap_noise, seed=0)

        assert len(list(generator)) == 2 + n_ops

    def test_haar_zero_ops_is_empty(self):
        assert list(RandomSptmHaarOperationsGenerator(wires=4, n_ops=0, seed=TEST_SEED)) == []

    def test_haar_rotation_blocks_sweep_neighbouring_pairs_of_wire_labels(self):
        ops = list(RandomSptmHaarOperationsGenerator(wires=[5, 3, 4, 6], n_ops=9, add_swap_noise=False, seed=0))
        rotations = ops[2:]

        assert [type(op) for op in rotations] == [SptmCompRzRz, SptmCompRyRy, SptmCompRzRz] * 3
        assert [tuple(op.wires.tolist()) for op in rotations[::3]] == [(3, 4), (4, 5), (5, 6)]
        assert tuple(ops[1].wires.tolist()) == (3, 4)

    def test_haar_swap_noise_on_neighbouring_pairs(self):
        ops = list(RandomSptmHaarOperationsGenerator(wires=[0, 1, 2, 5, 6], n_ops=60, seed=TEST_SEED))
        swaps = [op for op in ops if isinstance(op, SptmCompZX)]

        assert len(swaps) > 0
        assert self.wire_pairs(swaps) <= {(0, 1), (1, 2), (5, 6)}

    def test_haar_same_seed_gives_same_circuit(self):
        first = list(RandomSptmHaarOperationsGenerator(wires=4, n_ops=12, seed=TEST_SEED))
        second = list(RandomSptmHaarOperationsGenerator(wires=4, n_ops=12, seed=TEST_SEED))
        other = list(RandomSptmHaarOperationsGenerator(wires=4, n_ops=12, seed=TEST_SEED + 1))

        for matrix, same in zip(self.sptm_matrices(first), self.sptm_matrices(second)):
            np.testing.assert_allclose(matrix, same, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        assert not all(
            np.allclose(matrix, different)
            for matrix, different in zip(self.sptm_matrices(first), self.sptm_matrices(other))
        )

    def test_haar_circuit_gen_without_rn_gen(self):
        generator = RandomSptmHaarOperationsGenerator(wires=3, n_ops=3, seed=TEST_SEED)
        ops = list(generator.haar_circuit_gen())

        assert isinstance(ops[0], SptmIdentity)
        assert len(ops) == 4

    def test_haar_runs_on_nif_device(self):
        generator = RandomSptmHaarOperationsGenerator(wires=4, seed=TEST_SEED)
        probs = mc.NIFDevice(wires=4).execute_generator(generator, output_type="probs", reset=True)

        np.testing.assert_allclose(float(qml.math.sum(probs)), 1.0, atol=ATOL_MATRIX_COMPARISON)

    def test_haar_use_cuda_moves_every_op(self, monkeypatch):
        # ``to_cuda`` itself is GPU-only; stub it so the use_cuda=True path runs on a CPU-only machine.
        moved = []

        def fake_to_cuda(self):
            moved.append(self)
            return self

        monkeypatch.setattr(SingleParticleTransitionMatrixOperation, "to_cuda", fake_to_cuda)
        ops = list(RandomSptmHaarOperationsGenerator(wires=4, n_ops=12, use_cuda=True, seed=TEST_SEED))

        assert moved == ops[1:]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
    @pytest.mark.parametrize(
        "make_generator",
        [
            lambda use_cuda: RandomSptmHaarOperationsGenerator(wires=4, seed=TEST_SEED, use_cuda=use_cuda),
            lambda use_cuda: RandomSptmOperationsGenerator(wires=4, n_ops=20, seed=TEST_SEED, use_cuda=use_cuda),
            lambda use_cuda: random_sptm_operations_generator(20, wires=4, seed=TEST_SEED, use_cuda=use_cuda),
        ],
    )
    def test_use_cuda_matches_cpu_on_nif_device(self, make_generator):
        gpu_probs = mc.NIFDevice(wires=4).execute_generator(make_generator(True), output_type="probs", reset=True)
        cpu_probs = mc.NIFDevice(wires=4).execute_generator(make_generator(False), output_type="probs", reset=True)

        assert gpu_probs.is_cuda
        np.testing.assert_allclose(
            qml.math.to_numpy(gpu_probs),
            qml.math.to_numpy(cpu_probs),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
