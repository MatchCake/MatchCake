import numpy as np
import pytest

from matchcake.operations import SptmFHH, fH
from tests.configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestSptmFHH:
    def test_matchgate_equal_to_sptm_fhh(self):
        matchgate = fH(wires=[0, 1])
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmFHH(wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_matchgate_equal_to_sptm_fhh_adjoint(self):
        matchgate = fH(wires=[0, 1])
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmFHH(wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
