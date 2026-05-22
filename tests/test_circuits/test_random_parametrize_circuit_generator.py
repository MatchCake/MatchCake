import pytest

from matchcake import MatchgateOperation
from matchcake.circuits.random_parametrize_circuit_generator import RandomParametrizeGenerator

from ..configs import TEST_SEED, set_seed


class TestRandomParametrizeGenerator:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_is_subclass_of_random_operations_generator(self):
        from matchcake.circuits.random_generator import RandomOperationsGenerator

        assert issubclass(RandomParametrizeGenerator, RandomOperationsGenerator)

    def test_init(self):
        gen = RandomParametrizeGenerator(wires=4, n_ops=2, op_types=[MatchgateOperation])
        assert gen is not None
        assert gen.n_ops == 2

    @pytest.mark.parametrize("n_ops", [0, 1, 3])
    def test_len(self, n_ops):
        gen = RandomParametrizeGenerator(wires=4, n_ops=n_ops, op_types=[MatchgateOperation])
        assert len(gen) == n_ops

    def test_iter_produces_operations(self):
        gen = RandomParametrizeGenerator(
            wires=4, n_ops=2, op_types=[MatchgateOperation], seed=TEST_SEED
        )
        ops = list(gen)
        assert len(ops) == 3
