import pytest

from matchcake.operations.fermionic_swap import CompZX, fswap_chain, fswap_chain_gen

from ..configs import TEST_SEED, set_seed


class TestFermionicSwap:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_fswap_chain_gen_ascending(self):
        ops = list(fswap_chain_gen([0, 3]))
        assert len(ops) == 3
        for op in ops:
            assert isinstance(op, CompZX)

    def test_fswap_chain_gen_descending(self):
        ops = list(fswap_chain_gen([3, 0]))
        assert len(ops) == 3

    def test_fswap_chain_returns_list(self):
        ops = fswap_chain([0, 2])
        assert isinstance(ops, list)
        assert len(ops) == 2

    def test_fswap_chain_single_step(self):
        ops = fswap_chain([0, 1])
        assert len(ops) == 1
        assert isinstance(ops[0], CompZX)

    @pytest.mark.parametrize("wires,expected_len", [([0, 4], 4), ([2, 5], 3)])
    def test_fswap_chain_length(self, wires, expected_len):
        ops = fswap_chain(wires)
        assert len(ops) == expected_len
