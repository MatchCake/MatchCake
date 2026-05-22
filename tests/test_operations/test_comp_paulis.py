import numpy as np
import pytest

from matchcake.operations.comp_paulis import CompPauli, CompXX, CompYY, CompZZ

from ..configs import TEST_SEED, set_seed


class TestCompPauli:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_random_returns_instance(self):
        op = CompXX.random(wires=[0, 1])
        assert op is not None

    def test_comp_xx_constructs(self):
        op = CompXX(wires=[0, 1])
        assert op is not None

    def test_comp_yy_constructs(self):
        op = CompYY(wires=[0, 1])
        assert op is not None

    def test_comp_zz_constructs(self):
        op = CompZZ(wires=[0, 1])
        assert op is not None

    def test_comp_xx_matrix_shape(self):
        op = CompXX(wires=[0, 1])
        mat = op.matrix()
        assert mat.shape == (4, 4)

    def test_comp_pauli_wrong_length_raises(self):
        with pytest.raises(ValueError):
            CompPauli(paulis=["X"], wires=[0, 1])

    def test_comp_pauli_new_with_valid_paulis(self):
        op = CompXX(wires=[0, 1])
        assert op.matrix().shape == (4, 4)
