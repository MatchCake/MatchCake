import numpy as np
import pytest

from matchcake.operations import CompHH, SptmCompHH
from tests.configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestSptmCompHH:
    def test_matchgate_equal_to_sptm_fhh(self):
        matchgate = CompHH(wires=[0, 1])
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmCompHH(wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_matchgate_equal_to_sptm_fhh_adjoint(self):
        matchgate = CompHH(wires=[0, 1])
        m_sptm = matchgate.single_particle_transition_matrix
        sptm = SptmCompHH(wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            m_sptm,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
