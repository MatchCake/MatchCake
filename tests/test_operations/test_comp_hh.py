import numpy as np
import pytest

from matchcake.operations import CompHH
from matchcake.utils import make_single_particle_transition_matrix_from_gate


class TestCompHH:
    def test_form(self):
        mg = CompHH(wires=[0, 1])
        mg_matrix = (1 / np.sqrt(2)) * np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, -1, 0],
                [1, 0, 0, -1],
            ]
        )
        np.testing.assert_allclose(mg.matrix(), mg_matrix)

    def test_to_sptm_operation(self):
        mg = CompHH(wires=[0, 1])
        sptm = mg.to_sptm_operation().matrix()
        true_sptm = make_single_particle_transition_matrix_from_gate(mg.matrix())
        np.testing.assert_allclose(sptm, true_sptm)

    def test_random(self):
        rn_mg = CompHH.random(wires=[0, 1])
        mg = CompHH(wires=[0, 1])
        np.testing.assert_allclose(rn_mg.matrix(), mg.matrix())

    def test_label(self):
        mg = CompHH(wires=[0, 1])
        assert isinstance(mg.label(), str)
