import numpy as np

from matchcake.operations import SptmIdentity
from tests.configs import ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON


class TestSptmIdentity:
    def test_matchgate_equal_to_sptm_identity(self):
        sptm = SptmIdentity(wires=[0, 1]).matrix()
        np.testing.assert_allclose(
            sptm,
            np.eye(4),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
