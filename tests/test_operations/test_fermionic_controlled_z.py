import pytest

from matchcake.operations.fermionic_controlled_z import FermionicControlledZ, fCZ

from ..configs import TEST_SEED, set_seed


class TestFermionicControlledZ:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_compute_decomposition_returns_four_ops(self):
        ops = FermionicControlledZ.compute_decomposition(wires=[0, 1])
        assert len(ops) == 4

    def test_fCZ_alias(self):
        assert fCZ is FermionicControlledZ

    def test_label_default(self):
        op = FermionicControlledZ(wires=[0, 1])
        lbl = op.label()
        assert isinstance(lbl, str)

    def test_label_with_base_label(self):
        op = FermionicControlledZ(wires=[0, 1])
        lbl = op.label(base_label="fCZ")
        assert lbl == "fCZ"
