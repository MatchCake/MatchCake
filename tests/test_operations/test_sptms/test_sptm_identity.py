import numpy as np

from matchcake.operations import SptmIdentity
from matchcake.operations.matchgate_identity import MatchgateIdentity
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

    def test_adjoint_returns_self_matrix(self):
        op = SptmIdentity(wires=[0, 1])
        adj = op.adjoint()
        np.testing.assert_allclose(
            adj.matrix(),
            op.matrix(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_to_matchgate_returns_identity(self):
        op = SptmIdentity(wires=[0, 1])
        mg = op.to_matchgate()
        assert isinstance(mg, MatchgateIdentity)
