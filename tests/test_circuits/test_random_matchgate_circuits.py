import pennylane as qml

from matchcake.circuits.random_matchgate_circuits import (
    RandomMatchgateHaarOperationsGenerator,
)

from ..configs import TEST_SEED, set_seed


class TestRandomMatchgateHaarOperationsGenerator:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_iter_zero_ops_is_empty(self):
        gen = RandomMatchgateHaarOperationsGenerator(wires=4, n_ops=0, seed=TEST_SEED)
        assert gen.n_ops == 0
        assert list(gen) == []

    def test_iter_yields_basis_state_then_ops(self):
        gen = RandomMatchgateHaarOperationsGenerator(wires=4, n_ops=6, seed=TEST_SEED)
        ops = list(gen)
        assert len(ops) > 1
        # First yielded op is always the initial BasisState preparation.
        assert isinstance(ops[0], qml.BasisState)
