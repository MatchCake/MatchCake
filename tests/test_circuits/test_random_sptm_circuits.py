from matchcake.circuits.random_sptm_circuits import random_sptm_operations_generator
from matchcake.operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)

from ..configs import TEST_SEED, set_seed


class TestRandomSptmOperationsGenerator:
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
